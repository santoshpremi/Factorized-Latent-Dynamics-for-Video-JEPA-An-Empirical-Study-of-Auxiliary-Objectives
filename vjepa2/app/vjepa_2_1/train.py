# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import gc
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from app.vjepa_2_1.models.utils.masks_dist import compute_mask_distance
from app.vjepa_2_1.models.utils.modules import Lambda_LinearWarmupHold
from app.vjepa_2_1.transforms import make_transforms
from app.vjepa_2_1.utils import (
    init_opt,
    init_video_model,
    load_checkpoint,
    normalize_nested,
)
from src.datasets.data_manager import init_data
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from torch.nn.parallel import DistributedDataParallel


log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
MAX_REPEAT_COUNTS = 10

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__, force=True)


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
    r_file = cfgs_meta.get("read_checkpoint", None)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    logger.info(f"LD_PRELOAD: {os.environ.get('LD_PRELOAD')}")
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MASK
    cfgs_mask = args.get("mask")

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    uniform_power = cfgs_model.get("uniform_power", False)
    use_mask_tokens = cfgs_model.get("use_mask_tokens", False)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    is_causal = cfgs_model.get("is_causal", False)
    pred_is_causal = cfgs_model.get("pred_is_causal", False)
    init_type = cfgs_model.get("init_type", "default")
    img_temporal_dim_size = cfgs_model.get("img_temporal_dim_size", None)
    n_registers = cfgs_model.get("n_registers", 0)
    has_cls_first = cfgs_model.get("has_cls_first", False)
    interpolate_rope = cfgs_model.get("interpolate_rope", False)
    lambda_value_img = cfgs_model.get("lambda_value_img", 0.0)
    lambda_value_vid = cfgs_model.get("lambda_value_vid", 0.0)
    n_registers_predictor = cfgs_model.get("n_registers_predictor", 0)
    lambda_progressive = cfgs_model.get("lambda_progressive", True)
    normalize_predictor = cfgs_model.get("normalize_predictor", False)
    modality_embedding = cfgs_model.get("modality_embedding", False)
    levels_predictor = cfgs_model.get("levels_predictor", 4)
    if model_name == "vit_base":
        embed_dim_encoder = 768
    elif model_name == "vit_large":
        embed_dim_encoder = 1024
    elif model_name == "vit_giant_xformers":
        embed_dim_encoder = 1408
    elif model_name == "vit_gigantic_xformers":
        embed_dim_encoder = 1664
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "videodataset")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights")
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    frame_sample_rate = cfgs_data.get("frame_sample_rate")
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    grid_size = crop_size // patch_size
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)

    # -- IMG DATA
    cfgs_img_data = args.get("img_data")
    img_rank_ratio = 0.25
    img_mask = None
    if cfgs_img_data is not None:
        img_dataset_type = cfgs_img_data.get("dataset_type", "imagenet")
        img_dataset_paths = cfgs_img_data.get("datasets", [])
        img_dataset_weights = cfgs_img_data.get("datasets_weights", [])
        img_dataset_fpcs = cfgs_img_data.get("dataset_fpcs")
        img_dataset_batch_size = cfgs_img_data.get("batch_size")
        img_rank_ratio = cfgs_img_data.get("rank_ratio", img_rank_ratio)
        img_num_workers = cfgs_img_data.get("num_workers", num_workers)

        img_mask = args.get("img_mask", img_mask)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")
    shift_by_n = cfgs_loss.get("shift_by_n")
    predict_all = cfgs_loss.get("predict_all", True)
    weight_distance_loss = cfgs_loss.get("weight_distance_loss", False)
    offset_context_loss = cfgs_loss.get("offset_context_loss", False)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    disable_ema = cfgs_opt.get("disable_ema", False)
    sigreg_coeff = cfgs_opt.get("sigreg_coeff", 0.0)
    kinematic_coeff = cfgs_opt.get("kinematic_coeff", 0.0)
    kinematic_type = cfgs_opt.get("kinematic_type", "l1")
    kinematic_split_ratio = cfgs_opt.get("kinematic_split_ratio", 1.0)
    kinematic_anneal = cfgs_opt.get("kinematic_anneal", False)
    
    hamiltonian_coeff = cfgs_opt.get("hamiltonian_coeff", 0.0)
    hamiltonian_dt = cfgs_opt.get("hamiltonian_dt", 0.1)
    
    velgate_coeff = cfgs_opt.get("velgate_coeff", 0.0)
    velgate_percentile = cfgs_opt.get("velgate_percentile", 0.5)

    # Factorized World-Model JEPA (FWM-JEPA): structurally separate the latent
    # space into Appearance (Z_app, first fwm_app_ratio*D channels) and
    # Dynamics (Z_dyn, remaining channels). Z_app is encouraged to be temporally
    # invariant; Z_app and Z_dyn are pushed toward orthogonality via cross
    # covariance. Both losses operate on the student z_enc reconstructed onto
    # the full spatiotemporal grid so gradients flow into the encoder.
    fwm_static_coeff = cfgs_opt.get("fwm_static_coeff", 0.0)
    fwm_orth_coeff = cfgs_opt.get("fwm_orth_coeff", 0.0)
    fwm_app_ratio = cfgs_opt.get("fwm_app_ratio", 0.5)

    # Delta-Prediction JEPA: predict the temporal CHANGE of representations
    # (delta_h(t) = h(t+1) - h(t)) from the encoder. Forces the encoder to
    # capture dynamics as a first-class quantity, which is critical for
    # fine-grained motion benchmarks (Diving-48, SSv2). Aligned with LeCun's
    # world-model framing of JEPA: a latent dynamics predictor.
    delta_coeff = cfgs_opt.get("delta_coeff", 0.0)

    # Hard-Region Weighted Loss (HW-JEPA): focal-loss-style up-weighting of
    # the prediction loss on tokens where the predictor currently makes the
    # largest errors. Online adaptive hard-example mining within JEPA. The
    # weights use detached errors and a softmax with temperature hw_temp; the
    # gradient still flows through the original (unweighted) prediction loss.
    hw_coeff = cfgs_opt.get("hw_coeff", 0.0)
    hw_temp = cfgs_opt.get("hw_temp", 1.0)

    # Action-Conditioned JEPA (AC-JEPA): predict per-token "action" signal
    # (RGB frame-to-frame delta in token space) from encoder features.  This
    # forces the encoder to produce features from which the inter-frame action
    # is decodable, directly addressing the action-conditioning gap between
    # V-JEPA and LeCun's H-JEPA framework (where the predictor takes
    # (state, action) -> next state).  We use the dual auxiliary-loss form:
    # h_action = action_head(z_enc(t)), target = mean(frame_{t+1}) - mean(frame_t)
    # over each tubelet+patch tile.  Loss is MSE on visible token pairs.
    ac_coeff = cfgs_opt.get("ac_coeff", 0.0)
    
    # DDP-JEPA (Dual-Head Predictor) / LD-JEPA parameters
    ld_coeff = cfgs_opt.get("ld_coeff", 0.0)
    ld_hidden_dim = cfgs_opt.get("ld_hidden_dim", 256)
    
    # Spectral-JEPA parameters
    spectral_coeff = cfgs_opt.get("spectral_coeff", 0.0)
    
    # LTC-JEPA parameters
    ltc_coeff = cfgs_opt.get("ltc_coeff", 0.0)
    ltc_margin = cfgs_opt.get("ltc_margin", 0.5)
    ac_target_dim = cfgs_opt.get("ac_target_dim", 3)
    ac_hidden_dim = cfgs_opt.get("ac_hidden_dim", 256)

    is_anneal = cfgs_opt.get("is_anneal", False)
    anneal_ckpt = cfgs_opt.get("anneal_ckpt", None)
    if is_anneal and anneal_ckpt is None:
        raise ValueError("Must specify anneal_ckpt if is_anneal is True")
    resume_anneal = cfgs_opt.get("resume_anneal", False) or (
        is_anneal and resume_preempt
    )
    ipe = cfgs_opt.get("ipe", None)
    ipe_scale = cfgs_opt.get("ipe_scale", 1.0)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    ema = cfgs_opt.get("ema")
    use_radamw = cfgs_opt.get("use_radamw", False)
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    loss_reg_std_mult = cfgs_opt.get("loss_reg_std_mult", None)
    loss_reg_num_tracking_steps = cfgs_opt.get("loss_reg_num_tracking_steps", 300)
    loss_reg_min_epoch = cfgs_opt.get("loss_reg_min_epoch", 50)
    if loss_reg_std_mult is not None:
        logger.info("Loss regulation activated")
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    data_world_size, data_rank = world_size, rank
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    img_world_size = 0

    # make adjustments to batch size for image data
    model_fpcs = dataset_fpcs
    model_cfgs_mask = cfgs_mask
    model_tubelet_size = tubelet_size
    if cfgs_img_data is not None:
        img_world_size = int(world_size * img_rank_ratio)
        num_video_ranks = world_size - img_world_size
        img_total_batch_size = img_dataset_batch_size * world_size
        video_total_batch_size = batch_size * world_size

        if img_total_batch_size % img_world_size != 0:
            raise ValueError(
                f"img_total_batch_size ({img_total_batch_size}) must be divisible by num_img_ranks ({img_world_size})"
            )
        if video_total_batch_size % num_video_ranks != 0:
            raise ValueError(
                f"video_total_batch_size ({video_total_batch_size}) must be divisible by num_video_ranks ({num_video_ranks})"
            )

        # img_dataset_batch_size = img_total_batch_size // img_world_size
        batch_size = video_total_batch_size // num_video_ranks

        if rank < int(world_size * img_rank_ratio):
            crop_size = cfgs_img_data.get("crop_size", 512)
            grid_size = crop_size // patch_size

        if rank < int(world_size * img_rank_ratio):
            logger.info(
                f"On rank {rank}, updating dataset with dataset type {img_dataset_type}"
            )
            if img_temporal_dim_size is not None:
                if img_dataset_fpcs[0] != 1:
                    raise NotImplementedError(
                        "Image loader only supports 1 frame per clip with img_temporal_dim_size=1"
                    )
                tubelet_size = 1
            else:
                tubelet_size = tubelet_size

            dataset_type = img_dataset_type
            dataset_paths = img_dataset_paths
            datasets_weights = img_dataset_weights
            dataset_fpcs = img_dataset_fpcs
            batch_size = img_dataset_batch_size
            num_workers = img_num_workers
            if img_mask is not None:
                logger.info("Using image mask")
                cfgs_mask = img_mask

            data_rank = rank
            data_world_size = img_world_size
            lambda_value = lambda_value_img  # We select a different lambda value depending on video vs. image
        else:
            data_rank = rank - img_world_size
            data_world_size = world_size - img_world_size
            lambda_value = lambda_value_vid  # We select a different lambda value depending on video vs. image

        logger.info(
            f"For rank {rank} with world size {world_size}, "
            f"we have total image batch size {img_total_batch_size}, total video batch size {video_total_batch_size}, "
            f"image ranks: {img_world_size}, video ranks: {num_video_ranks}, "
            f"using the following params: "
            f"dataset_type: {dataset_type}, "
            f"dataset_paths: {dataset_paths}, "
            f"datasets_weights: {datasets_weights}, "
            f"dataset_fpcs: {dataset_fpcs}, "
            f"batch_size: {batch_size}, "
            f"num_workers: {num_workers}, "
            f"data_rank: {data_rank}, "
            f"data_world_size: {data_world_size}"
            f"lambda_value for the context loss: {lambda_value}"
        )
    else:
        lambda_value = lambda_value_vid

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        # With torchrun, CUDA_VISIBLE_DEVICES is often already set to a single GPU per process
        # or we just use cuda:0 for the visible device
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    if sigreg_coeff > 0.0:
        from lejepa.univariate import EppsPulley
        from lejepa.multivariate import SlicingUnivariateTest
        univariate_test = EppsPulley(t_max=3, n_points=17, integration="trapezoid").to(device)
        sigreg_loss_fn = SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=1024,
        ).to(device)
    else:
        sigreg_loss_fn = None

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_file = "latest.pth.tar"
    latest_path = os.path.join(folder, latest_file)

    load_path = None
    if load_model:
        if is_anneal:
            if os.path.exists(latest_path) and resume_anneal:
                load_path = latest_path
            else:
                load_path = anneal_ckpt
                resume_anneal = False
        else:
            load_path = r_file if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
    )

    # -- init model
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=int(len(model_cfgs_mask) * len(model_fpcs)),
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=model_tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        is_causal=is_causal,
        pred_is_causal=pred_is_causal,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
        return_all_tokens=predict_all,
        chop_last_n_tokens=shift_by_n,
        init_type=init_type,
        img_temporal_dim_size=img_temporal_dim_size,
        n_registers=n_registers,
        n_registers_predictor=n_registers_predictor,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
    )
    target_encoder = copy.deepcopy(encoder)

    if hamiltonian_coeff > 0.0:
        from app.vjepa_2_1.models.hamiltonian import HamiltonianNN
        # If levels_predictor is 4, the target encoder outputs 4 * embed_dim_encoder
        # We will apply Hamiltonian loss to the full concatenated dimension.
        full_dim = embed_dim_encoder * levels_predictor
        hamiltonian_net = HamiltonianNN(dim=full_dim // 2, hidden_dim=256).to(device)
        hamiltonian_net = DistributedDataParallel(hamiltonian_net, static_graph=True)
    else:
        hamiltonian_net = None

    if ac_coeff > 0.0:
        # AC-JEPA action head: maps encoder features at time t to predicted
        # per-token action (frame delta) between t and t+1.  Small 2-layer
        # MLP keeps capacity bounded so the encoder is forced to produce
        # action-decodable representations rather than memorizing through a
        # large head.
        full_dim_action = embed_dim_encoder * levels_predictor
        
        # FAC-JEPA: If FWM is active, action_head only takes Z_dyn
        fwm_static_coeff = cfgs_opt.get("fwm_static_coeff", 0.0)
        fwm_orth_coeff = cfgs_opt.get("fwm_orth_coeff", 0.0)
        if fwm_static_coeff > 0.0 or fwm_orth_coeff > 0.0:
            fwm_app_ratio = cfgs_opt.get("fwm_app_ratio", 0.5)
            D_app = max(1, int(full_dim_action * fwm_app_ratio))
            full_dim_action = full_dim_action - D_app
            logger.info(f"FAC-JEPA active: action_head input dim reduced to {full_dim_action} (Z_dyn)")

        action_head = torch.nn.Sequential(
            torch.nn.Linear(full_dim_action, ac_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(ac_hidden_dim, ac_target_dim),
        ).to(device)
        action_head = DistributedDataParallel(action_head, static_graph=True)
    else:
        action_head = None

    if ld_coeff > 0.0:
        # LD-JEPA (Latent Dynamics JEPA) / DDP-JEPA
        # Predicts h_{t+1} - h_t from z_enc_{t}
        full_dim_ld = embed_dim_encoder * levels_predictor
        
        # FWM-LD: If FWM is active, dyn_head only takes Z_dyn
        fwm_static_coeff = cfgs_opt.get("fwm_static_coeff", 0.0)
        fwm_orth_coeff = cfgs_opt.get("fwm_orth_coeff", 0.0)
        if fwm_static_coeff > 0.0 or fwm_orth_coeff > 0.0:
            fwm_app_ratio = cfgs_opt.get("fwm_app_ratio", 0.5)
            D_app = max(1, int(full_dim_ld * fwm_app_ratio))
            full_dim_ld_in = full_dim_ld - D_app
            logger.info(f"FWM-LD active: dyn_head input dim reduced to {full_dim_ld_in} (Z_dyn)")
        else:
            full_dim_ld_in = full_dim_ld

        dyn_head = torch.nn.Sequential(
            torch.nn.Linear(full_dim_ld_in, ld_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(ld_hidden_dim, full_dim_ld),
        ).to(device)
        dyn_head = DistributedDataParallel(dyn_head, static_graph=True)
    else:
        dyn_head = None

    if compile_model:
        logger.info("Compiling encoder, target_encoder, and predictor.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        target_encoder.compile()
        predictor.compile()

    mask_collator = MaskCollator(
        cfgs_mask=cfgs_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )

    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    # -- init data-loaders/samplers
    (unsupervised_loader, unsupervised_sampler) = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=True,
        # clip_len=clip_len,
        dataset_fpcs=dataset_fpcs,
        frame_sample_rate=frame_sample_rate,
        fps=fps,
        transform=transform,
        rank=data_rank,
        world_size=data_world_size,
        datasets_weights=datasets_weights,
        collator=mask_collator,
        num_workers=num_workers,
        pin_mem=pin_mem,
        log_dir=None,
    )
    try:
        _dlen = len(unsupervised_loader)
    except Exception:
        try:
            _dlen = unsupervised_loader.num_batches
        except Exception:
            _dlen = -1
    if ipe is None:
        ipe = _dlen
    logger.info(f"Using batch size of {batch_size}, fpcs of {dataset_fpcs}")
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # zizi

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        is_anneal=is_anneal,
        encoder=encoder,
        predictor=predictor,
        use_radamw=use_radamw,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
        hamiltonian_net=hamiltonian_net,
        action_head=action_head,
        dyn_head=dyn_head,
    )
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(
        predictor, static_graph=False, find_unused_parameters=True
    )
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs) + 1)
    )
    lambda_sched = Lambda_LinearWarmupHold(lambda_value=lambda_value)

    start_epoch = 0
    # -- load training checkpoint
    print("Loadind checkpoint from: ", load_path)
    if load_path is not None:
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
            hamiltonian_net,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
            is_anneal=is_anneal and not resume_anneal,
            hamiltonian_net=hamiltonian_net,
        )
        if not is_anneal or resume_anneal:
            for _ in range(start_epoch * ipe):
                scheduler.step()
                wd_scheduler.step()
                next(momentum_scheduler)
                mask_collator.step()

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "hamiltonian_net": hamiltonian_net.state_dict() if hamiltonian_net is not None else None,
            "action_head": action_head.state_dict() if action_head is not None else None,
            "dyn_head": dyn_head.state_dict() if dyn_head is not None else None,
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    logger.info("Initializing loader...")
    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")

        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    trailing_losses = []
    step_count = 0

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        loss_meter = AverageMeter()
        mask_meters = {fpc: AverageMeter() for fpc in dataset_fpcs}
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    logger.info("Exhausted data loaders. Refreshing...")
                    if "airstore" in dataset_type.lower():
                        unsupervised_sampler.increase_epoch()
                    else:
                        unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(
                            f"Encountered exception when loading data (num retries {iter_retries}):\n{e}"
                        )
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        raise RuntimeError(
                            f"Exceeded max retries ({NUM_RETRIES}) when loading data."
                        ) from e

            for _fpc_sample in sample:
                bs, fpc = _fpc_sample[0][-1][0].size()
                mask_meters[fpc].update(bs / batch_size)

            def load_clips():
                all_clips, all_masks_enc, all_masks_pred = [], [], []
                for fpc_sample in sample:
                    udata, masks_enc, masks_pred = fpc_sample
                    all_clips += [udata[0][0].to(device, non_blocking=True)]
                    all_masks_enc += [
                        [m.to(device, non_blocking=True) for m in masks_enc]
                    ]
                    all_masks_pred += [
                        [m.to(device, non_blocking=True) for m in masks_pred]
                    ]
                return all_clips, all_masks_enc, all_masks_pred

            clips, masks_enc, masks_pred = load_clips()
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target(c, embed_dim=embed_dim_encoder):
                    if disable_ema:
                        h = encoder(c, gram_mode=False, training_mode=True)
                    else:
                        with torch.no_grad():
                            h = target_encoder(c, gram_mode=False, training_mode=True)
                    new_h = []
                    for hi in h:
                        if levels_predictor > 1:
                            hi_0 = F.layer_norm(hi[:, :, :embed_dim], (embed_dim,))
                            hi_1 = F.layer_norm(
                                hi[:, :, embed_dim : embed_dim * 2],
                                (embed_dim,),
                            )
                            hi_2 = F.layer_norm(
                                hi[:, :, embed_dim * 2 : embed_dim * 3],
                                (embed_dim,),
                            )
                            hi_3 = F.layer_norm(hi[:, :, -embed_dim:], (embed_dim,))
                            hi_norm = torch.cat([hi_0, hi_1, hi_2, hi_3], dim=2)
                            new_h.append(hi_norm)
                        else:
                            new_h.append(F.layer_norm(hi, (hi.size(-1),)))
                    return new_h

                def forward_context(clips, embed_dim=embed_dim_encoder):
                    modality = "video"
                    if img_temporal_dim_size is not None:
                        if clips[0].shape[2] == img_temporal_dim_size:
                            modality = "image"
                    z = encoder(clips, masks_enc, gram_mode=False, training_mode=True)
                    z_pred, z_context = predictor(
                        z, masks_enc, masks_pred, mod=modality
                    )
                    if normalize_predictor:
                        z_pred = normalize_nested(z_pred, embed_dim)

                        if predict_all:
                            z_context = normalize_nested(z_context, embed_dim)
                    return z_pred, z_context, z

                def loss_fn(z, h, masks_to_apply, cls_loss, d_weights):
                    if cls_loss:
                        h_cls = [hi[:, 0].unsqueeze(1) for hi in h]
                        h = [
                            apply_masks(hi[:, 1:], mi, concat=False)
                            for hi, mi in zip(h, masks_to_apply)
                        ]
                        loss, n = 0, 0
                        for zi, hi, hi_cls in zip(z, h, h_cls):
                            for zij, hij in zip(zi, hi):
                                h_term = torch.cat([hi_cls, hij], dim=1)
                                loss += (
                                    torch.mean(torch.abs(zij - h_term) ** loss_exp)
                                    / loss_exp
                                )
                                n += 1

                        loss /= n
                        return loss
                    else:
                        h = [
                            apply_masks(hi, mi, concat=False)
                            for hi, mi in zip(h, masks_to_apply)
                        ]

                        if d_weights is not None:
                            loss, n = 0, 0
                            for zi, hi, d_i in zip(z, h, d_weights):
                                for zij, hij, d_ij in zip(zi, hi, d_i):
                                    loss_n = torch.abs(zij - hij) ** loss_exp * (
                                        1 / d_ij.unsqueeze(2)
                                    )
                                    loss += torch.mean(loss_n) / loss_exp
                                    n += 1
                            loss /= n
                            return loss
                        else:
                            loss, n = 0, 0
                            for zi, hi in zip(z, h):
                                for zij, hij in zip(zi, hi):
                                    loss += (
                                        torch.mean(torch.abs(zij - hij) ** loss_exp)
                                        / loss_exp
                                    )
                                    n += 1
                            loss /= n
                            return loss

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    h = forward_target(clips)
                    z_pred, z_context, z_enc = forward_context(clips)
                    loss = 0
                    loss_pred = loss_fn(
                        z_pred, h, masks_pred, cls_loss=has_cls_first, d_weights=None
                    )
                    loss += loss_pred

                    # Hard-Region Weighted Loss (HW-JEPA): up-weight prediction
                    # loss on tokens with the largest prediction error.  Weights
                    # use detached errors via softmax with temperature hw_temp,
                    # then are renormalized so their mean is 1, so the magnitude
                    # of the auxiliary loss is comparable to loss_pred.  This is
                    # an additive auxiliary term scaled by hw_coeff -- it does
                    # not alter loss_pred itself.
                    if hw_coeff > 0.0 and not has_cls_first:
                        hw_loss = 0.0
                        n_hw = 0
                        h_for_pred = [
                            apply_masks(hi, mi, concat=False)
                            for hi, mi in zip(h, masks_pred)
                        ]
                        for zi, hi in zip(z_pred, h_for_pred):
                            for zij, hij in zip(zi, hi):
                                with torch.no_grad():
                                    err = (
                                        torch.abs(zij - hij) ** loss_exp
                                    ) / loss_exp
                                    err_per_tok = err.mean(dim=-1)
                                    err_f = err_per_tok.float()
                                    K_tok = err_f.shape[-1]
                                    if K_tok < 2:
                                        continue
                                    w = torch.softmax(
                                        err_f / max(hw_temp, 1.0e-6), dim=-1
                                    )
                                    w = w * float(K_tok)
                                    w = w.to(zij.dtype).unsqueeze(-1)
                                w_loss = (
                                    torch.abs(zij - hij) ** loss_exp
                                ) / loss_exp
                                hw_loss = hw_loss + (w_loss * w).mean()
                                n_hw += 1
                        if n_hw > 0:
                            hw_loss = hw_loss / n_hw
                            loss = loss + hw_coeff * hw_loss

                    # Context loss
                    if predict_all:
                        distance_weights = compute_mask_distance(
                            masks_pred, masks_enc, grid_size, offset_context_loss
                        )
                        if weight_distance_loss:
                            d_weights = distance_weights
                        else:
                            d_weights = None
                        loss_context = loss_fn(
                            z_context, h, masks_enc, cls_loss=False, d_weights=d_weights
                        )
                        if lambda_progressive:
                            lambda_value_step = lambda_sched.value(epoch * ipe + itr)
                        else:
                            lambda_value_step = lambda_value
                        loss += loss_context * lambda_value_step
                        
                    # SIGReg loss
                    if sigreg_coeff > 0.0 and sigreg_loss_fn is not None:
                        # z_enc is a list of lists of tensors [batch, tokens, dim]
                        z_flat_list = []
                        for z_seq in z_enc:
                            for zi in z_seq:
                                z_flat_list.append(zi.reshape(-1, zi.shape[-1]))
                        z_flat = torch.cat(z_flat_list, dim=0)
                        sigreg_loss = sigreg_loss_fn(z_flat)
                        loss += sigreg_coeff * sigreg_loss

                    # Hamiltonian-JEPA Loss
                    if hamiltonian_coeff > 0.0 and hamiltonian_net is not None:
                        from app.vjepa_2_1.models.hamiltonian import symplectic_euler_step
                        hamiltonian_loss = 0.0
                        n_hamiltonian = 0
                        for i, (z_seq, hi, m_seq) in enumerate(zip(z_enc, h, masks_enc)):
                            for zi, mi in zip(z_seq, m_seq):
                                B, K, D = zi.shape
                                
                                T_tokens = clips[i].shape[2] // model_tubelet_size
                                H_tokens = clips[i].shape[3] // patch_size
                                W_tokens = clips[i].shape[4] // patch_size
                                N = T_tokens * H_tokens * W_tokens
                                
                                if hi.shape[1] == N and T_tokens >= 2:
                                    # Reconstruct full grid for zi
                                    full_z = torch.zeros(B, N, D, device=zi.device, dtype=zi.dtype)
                                    full_z.scatter_(1, mi.unsqueeze(-1).expand(-1, -1, D), zi)
                                    full_z = full_z.view(B, T_tokens, H_tokens, W_tokens, D)
                                    
                                    # Reshape hi
                                    hi_reshaped = hi.view(B, T_tokens, H_tokens, W_tokens, D)
                                    
                                    # Valid mask
                                    valid_mask = torch.zeros(B, N, device=zi.device, dtype=torch.bool)
                                    valid_mask.scatter_(1, mi, True)
                                    valid_mask = valid_mask.view(B, T_tokens, H_tokens, W_tokens)
                                    
                                    # Valid pairs (t and t+1)
                                    valid_pairs = valid_mask[:, :-1] & valid_mask[:, 1:] # [B, T-1, H, W]
                                    
                                    if valid_pairs.any():
                                        # Split student's context representation into q and p
                                        q = full_z[:, :-1, :, :, :D//2] # [B, T-1, H, W, D/2]
                                        p = full_z[:, :-1, :, :, D//2:] # [B, T-1, H, W, D/2]
                                        
                                        # Target for next timestep comes from the stable EMA target encoder
                                        q_target = hi_reshaped[:, 1:, :, :, :D//2].detach()
                                        p_target = hi_reshaped[:, 1:, :, :, D//2:].detach()
                                        
                                        # Flatten spatial dimensions for Hamiltonian net
                                        q_flat = q.reshape(B, (T_tokens-1) * H_tokens * W_tokens, D//2)
                                        p_flat = p.reshape(B, (T_tokens-1) * H_tokens * W_tokens, D//2)
                                        
                                        # Evolve student's q and p using Hamiltonian Neural Network
                                        q_pred_flat, p_pred_flat = symplectic_euler_step(q_flat, p_flat, hamiltonian_net, dt=hamiltonian_dt)
                                        
                                        q_pred = q_pred_flat.view(B, T_tokens-1, H_tokens, W_tokens, D//2)
                                        p_pred = p_pred_flat.view(B, T_tokens-1, H_tokens, W_tokens, D//2)
                                        
                                        # Loss is the prediction error of the Hamiltonian evolution against the target
                                        loss_q = torch.abs(q_pred - q_target) ** loss_exp / loss_exp
                                        loss_p = torch.abs(p_pred - p_target) ** loss_exp / loss_exp
                                        
                                        # Mask and mean
                                        h_loss_q = loss_q[valid_pairs].mean()
                                        h_loss_p = loss_p[valid_pairs].mean()
                                        
                                        hamiltonian_loss += (h_loss_q + h_loss_p)
                                        n_hamiltonian += 1
                                else:
                                    logger.warning(f"Hamiltonian-JEPA: hi.shape[1] ({hi.shape[1]}) != N ({N}) or T_tokens ({T_tokens}) < 2")
                        
                        if n_hamiltonian > 0:
                            hamiltonian_loss /= n_hamiltonian
                            loss += hamiltonian_coeff * hamiltonian_loss

                    # Velocity-Gated Kinematics Loss
                    if velgate_coeff > 0.0:
                        velgate_loss = 0.0
                        n_velgate = 0
                        for i, (z_seq, hi, m_seq) in enumerate(zip(z_enc, h, masks_enc)):
                            for zi, mi in zip(z_seq, m_seq):
                                B, K, D = zi.shape
                                
                                T_tokens = clips[i].shape[2] // model_tubelet_size
                                H_tokens = clips[i].shape[3] // patch_size
                                W_tokens = clips[i].shape[4] // patch_size
                                N = T_tokens * H_tokens * W_tokens
                                
                                if hi.shape[1] == N and T_tokens >= 2:
                                    # Reconstruct full grid for zi
                                    full_z = torch.zeros(B, N, D, device=zi.device, dtype=zi.dtype)
                                    full_z.scatter_(1, mi.unsqueeze(-1).expand(-1, -1, D), zi)
                                    full_z = full_z.view(B, T_tokens, H_tokens, W_tokens, D)
                                    
                                    # Reshape hi
                                    hi_reshaped = hi.view(B, T_tokens, H_tokens, W_tokens, D)
                                    
                                    # Valid mask
                                    valid_mask = torch.zeros(B, N, device=zi.device, dtype=torch.bool)
                                    valid_mask.scatter_(1, mi, True)
                                    valid_mask = valid_mask.view(B, T_tokens, H_tokens, W_tokens)
                                    
                                    # Valid pairs (t and t+1)
                                    valid_pairs = valid_mask[:, :-1] & valid_mask[:, 1:] # [B, T-1, H, W]
                                    
                                    if valid_pairs.any():
                                        # Compute velocity on stable target encoder features
                                        delta_h = hi_reshaped[:, 1:] - hi_reshaped[:, :-1]
                                        velocity = torch.norm(delta_h, dim=-1) # [B, T-1, H, W]
                                        
                                        # Compute threshold per-video (B) and per-frame (T-1)
                                        vel_flat = velocity.detach().view(B, T_tokens - 1, -1) # [B, T-1, HW]
                                        vel_threshold = torch.quantile(vel_flat, velgate_percentile, dim=-1, keepdim=True) # [B, T-1, 1]
                                        vel_threshold = vel_threshold.unsqueeze(-1) # Broadcast to [B, T-1, 1, 1]
                                        
                                        # Gate: 1.0 if velocity <= threshold (background), 0.0 otherwise (foreground)
                                        gate = (velocity.detach() <= vel_threshold).float() # [B, T-1, H, W]
                                        
                                        # Apply loss to student's context representations
                                        delta_z = full_z[:, 1:] - full_z[:, :-1] # [B, T-1, H, W, D]
                                        
                                        v_loss = gate.unsqueeze(-1) * torch.abs(delta_z)
                                        velgate_loss += v_loss[valid_pairs].mean()
                                        n_velgate += 1
                                else:
                                    logger.warning(f"VelGate-JEPA: hi.shape[1] ({hi.shape[1]}) != N ({N}) or T_tokens ({T_tokens}) < 2")
                        
                        if n_velgate > 0:
                            velgate_loss /= n_velgate
                            loss += velgate_coeff * velgate_loss

                    # FWM-JEPA: Factorized World-Model JEPA Loss
                    # L_static: temporal invariance of Z_app (appearance subspace)
                    # L_orth:   cross-covariance between Z_app and Z_dyn -> 0
                    if fwm_static_coeff > 0.0 or fwm_orth_coeff > 0.0:
                        fwm_static_loss = 0.0
                        fwm_orth_loss = 0.0
                        n_fwm_static = 0
                        n_fwm_orth = 0

                        for i, (z_seq, m_seq) in enumerate(zip(z_enc, masks_enc)):
                            for zi, mi in zip(z_seq, m_seq):
                                B, K, D = zi.shape
                                D_app = max(1, int(D * fwm_app_ratio))

                                T_tokens = clips[i].shape[2] // model_tubelet_size
                                H_tokens = clips[i].shape[3] // patch_size
                                W_tokens = clips[i].shape[4] // patch_size
                                N = T_tokens * H_tokens * W_tokens

                                # Image clips (T_tokens < 2) carry no temporal
                                # signal for the static loss; skip them safely.
                                if T_tokens < 2:
                                    continue

                                if mi.shape[1] > N:
                                    continue

                                # Reconstruct full spatiotemporal grid from the
                                # context tokens so gradients flow into encoder.
                                full_z = torch.zeros(
                                    B, N, D, device=zi.device, dtype=zi.dtype
                                )
                                full_z.scatter_(
                                    1,
                                    mi.unsqueeze(-1).expand(-1, -1, D),
                                    zi,
                                )
                                full_z = full_z.view(
                                    B, T_tokens, H_tokens, W_tokens, D
                                )

                                valid_mask = torch.zeros(
                                    B, N, device=zi.device, dtype=torch.bool
                                )
                                valid_mask.scatter_(1, mi, True)
                                valid_mask = valid_mask.view(
                                    B, T_tokens, H_tokens, W_tokens
                                )

                                z_app = full_z[..., :D_app]
                                z_dyn = full_z[..., D_app:]

                                # L_static: encourage Z_app(t+1) ~= Z_app(t)
                                if fwm_static_coeff > 0.0:
                                    valid_pairs = (
                                        valid_mask[:, :-1] & valid_mask[:, 1:]
                                    )
                                    if valid_pairs.any():
                                        delta_app = (
                                            z_app[:, 1:] - z_app[:, :-1]
                                        )
                                        static_term = torch.abs(delta_app)
                                        # mean over valid pair locations and
                                        # over channel dimension
                                        fwm_static_loss = (
                                            fwm_static_loss
                                            + static_term[valid_pairs].mean()
                                        )
                                        n_fwm_static += 1

                                # L_orth: cross-covariance between Z_app and Z_dyn
                                if fwm_orth_coeff > 0.0 and valid_mask.any():
                                    z_app_v = z_app[valid_mask]
                                    z_dyn_v = z_dyn[valid_mask]
                                    if z_app_v.shape[0] > 1:
                                        z_app_c = (
                                            z_app_v
                                            - z_app_v.mean(dim=0, keepdim=True)
                                        )
                                        z_dyn_c = (
                                            z_dyn_v
                                            - z_dyn_v.mean(dim=0, keepdim=True)
                                        )
                                        denom = max(z_app_c.shape[0] - 1, 1)
                                        # cast to float32 for a stable matmul
                                        # under bfloat16 autocast
                                        cross_cov = (
                                            z_app_c.float().t()
                                            @ z_dyn_c.float()
                                        ) / float(denom)
                                        fwm_orth_loss = (
                                            fwm_orth_loss
                                            + (cross_cov ** 2).mean().to(zi.dtype)
                                        )
                                        n_fwm_orth += 1

                        if fwm_static_coeff > 0.0 and n_fwm_static > 0:
                            fwm_static_loss = fwm_static_loss / n_fwm_static
                            loss = loss + fwm_static_coeff * fwm_static_loss

                        if fwm_orth_coeff > 0.0 and n_fwm_orth > 0:
                            fwm_orth_loss = fwm_orth_loss / n_fwm_orth
                            loss = loss + fwm_orth_coeff * fwm_orth_loss

                    # Delta-Prediction JEPA: align student temporal differences
                    # with target temporal differences.  L_delta = L1(student_z(t+1)
                    # - student_z(t),  detached_h(t+1) - detached_h(t))  over
                    # spatiotemporal positions where both consecutive tokens are
                    # in the encoder's context (so gradients flow into encoder).
                    if delta_coeff > 0.0 or spectral_coeff > 0.0 or ltc_coeff > 0.0 or ld_coeff > 0.0:
                        delta_loss = 0.0
                        spectral_loss = 0.0
                        ltc_loss = 0.0
                        ld_loss = 0.0
                        n_delta = 0
                        n_spectral = 0
                        n_ltc = 0
                        n_ld = 0
                        for i, (z_seq, hi, m_seq) in enumerate(
                            zip(z_enc, h, masks_enc)
                        ):
                            for zi, mi in zip(z_seq, m_seq):
                                B, K, D = zi.shape
                                T_tokens = (
                                    clips[i].shape[2] // model_tubelet_size
                                )
                                H_tokens = clips[i].shape[3] // patch_size
                                W_tokens = clips[i].shape[4] // patch_size
                                N = T_tokens * H_tokens * W_tokens
                                if T_tokens < 2:
                                    continue
                                if mi.shape[1] > N or hi.shape[1] != N:
                                    continue
                                # Reconstruct full grid for student z
                                full_z = torch.zeros(
                                    B, N, D, device=zi.device, dtype=zi.dtype
                                )
                                full_z.scatter_(
                                    1,
                                    mi.unsqueeze(-1).expand(-1, -1, D),
                                    zi,
                                )
                                full_z = full_z.view(
                                    B, T_tokens, H_tokens, W_tokens, D
                                )
                                full_h = hi.detach().view(
                                    B, T_tokens, H_tokens, W_tokens, D
                                )
                                valid_mask = torch.zeros(
                                    B, N, device=zi.device, dtype=torch.bool
                                )
                                valid_mask.scatter_(1, mi, True)
                                valid_mask = valid_mask.view(
                                    B, T_tokens, H_tokens, W_tokens
                                )
                                valid_pairs = (
                                    valid_mask[:, :-1] & valid_mask[:, 1:]
                                )
                                if valid_pairs.any():
                                    if delta_coeff > 0.0:
                                        delta_z = full_z[:, 1:] - full_z[:, :-1]
                                        delta_h = full_h[:, 1:] - full_h[:, :-1]
                                        delta_err = torch.abs(delta_z - delta_h)
                                        delta_loss = (
                                            delta_loss
                                            + delta_err[valid_pairs].mean()
                                        )
                                        n_delta += 1
                                    
                                    if ld_coeff > 0.0 and dyn_head is not None:
                                        # LD-JEPA: dyn_head(z_t) -> h_{t+1} - h_t
                                        z_for_dyn = full_z[:, :-1]
                                        
                                        # FWM-LD: If FWM is active, only use Z_dyn
                                        if fwm_static_coeff > 0.0 or fwm_orth_coeff > 0.0:
                                            D_app = max(1, int(D * fwm_app_ratio))
                                            z_for_dyn = z_for_dyn[..., D_app:]
                                            
                                        pred_delta_h = dyn_head(z_for_dyn)
                                        target_delta_h = full_h[:, 1:] - full_h[:, :-1]
                                        ld_err = torch.abs(pred_delta_h - target_delta_h)
                                        
                                        if hw_coeff > 0.0:
                                            with torch.no_grad():
                                                err_valid = ld_err[valid_pairs]
                                                if err_valid.numel() > 0:
                                                    err_mean = err_valid.mean(dim=-1)
                                                    w = torch.nn.functional.softmax(err_mean / hw_temp, dim=-1)
                                                    w = w * w.shape[-1]
                                                    ld_loss = ld_loss + (err_valid * w.unsqueeze(-1)).mean()
                                        else:
                                            ld_loss = ld_loss + ld_err[valid_pairs].mean()
                                        n_ld += 1
                                        
                                    if ltc_coeff > 0.0:
                                        # LTC-JEPA: Contrastive loss on temporal sequence
                                        # z_t should be close to h_t, far from h_{t+1}
                                        z_t = full_z[:, :-1][valid_pairs]
                                        h_t = full_h[:, :-1][valid_pairs]
                                        h_t_next = full_h[:, 1:][valid_pairs]
                                        
                                        pos_sim = torch.nn.functional.cosine_similarity(z_t, h_t)
                                        neg_sim = torch.nn.functional.cosine_similarity(z_t, h_t_next)
                                        
                                        # Margin loss: max(0, margin - pos_sim + neg_sim)
                                        ltc_err = torch.clamp(ltc_margin - pos_sim + neg_sim, min=0.0)
                                        ltc_loss = ltc_loss + ltc_err.mean()
                                        n_ltc += 1
                                        
                                if spectral_coeff > 0.0 and T_tokens >= 4:
                                    # Spectral-JEPA: FFT along temporal dimension
                                    # We only compute this if we have enough temporal tokens
                                    # Valid mask needs to be true for the whole temporal sequence for a spatial patch
                                    spatial_valid = valid_mask.all(dim=1) # [B, H, W]
                                    if spatial_valid.any():
                                        # Extract valid sequences: [N_valid_spatial, T, D]
                                        z_seqs = full_z.permute(0, 2, 3, 1, 4)[spatial_valid]
                                        h_seqs = full_h.permute(0, 2, 3, 1, 4)[spatial_valid]
                                        
                                        # Apply FFT
                                        fft_z = torch.fft.rfft(z_seqs, dim=1) # [N, T//2 + 1, D]
                                        fft_h = torch.fft.rfft(h_seqs, dim=1)
                                        
                                        # Weight high frequencies more (linear from 0.1 to 1.0)
                                        n_freqs = fft_z.shape[1]
                                        freq_weights = torch.linspace(0.1, 1.0, n_freqs, device=zi.device).view(1, -1, 1)
                                        
                                        # MSE in frequency domain
                                        spectral_err = torch.abs(fft_z * freq_weights - fft_h * freq_weights)
                                        spectral_loss = spectral_loss + spectral_err.mean()
                                        n_spectral += 1
                                        
                        if delta_coeff > 0.0 and n_delta > 0:
                            delta_loss = delta_loss / n_delta
                            loss = loss + delta_coeff * delta_loss
                            
                        if ld_coeff > 0.0 and n_ld > 0:
                            ld_loss = ld_loss / n_ld
                            loss = loss + ld_coeff * ld_loss
                            
                        if ltc_coeff > 0.0 and n_ltc > 0:
                            ltc_loss = ltc_loss / n_ltc
                            loss = loss + ltc_coeff * ltc_loss
                            
                        if spectral_coeff > 0.0 and n_spectral > 0:
                            spectral_loss = spectral_loss / n_spectral
                            loss = loss + spectral_coeff * spectral_loss

                    # AC-JEPA: Action-Conditioned JEPA auxiliary loss.
                    # Predict per-token RGB frame delta (the "action") from
                    # encoder features at time t. Target = mean(frame_{t+1}) -
                    # mean(frame_t), pooled over each tubelet+patch tile.
                    # Forces encoder to capture motion/action information,
                    # closing the action-conditioning gap between V-JEPA and
                    # LeCun's H-JEPA framework.
                    if ac_coeff > 0.0 and action_head is not None:
                        ac_loss = 0.0
                        n_ac = 0
                        for i, (z_seq, m_seq) in enumerate(
                            zip(z_enc, masks_enc)
                        ):
                            clip_i = clips[i]
                            B_clip, C_clip, T_clip, H_clip, W_clip = (
                                clip_i.shape
                            )
                            T_tokens = T_clip // model_tubelet_size
                            H_tokens = H_clip // patch_size
                            W_tokens = W_clip // patch_size
                            N = T_tokens * H_tokens * W_tokens
                            if T_tokens < 2:
                                continue
                            # Per-token RGB mean over (tubelet, patch_h, patch_w)
                            # -> [B, C, T_tokens, H_tokens, W_tokens]
                            with torch.no_grad():
                                patches = clip_i.float().reshape(
                                    B_clip,
                                    C_clip,
                                    T_tokens,
                                    model_tubelet_size,
                                    H_tokens,
                                    patch_size,
                                    W_tokens,
                                    patch_size,
                                )
                                patch_means = patches.mean(dim=(3, 5, 7))
                                # Action = patch_mean(t+1) - patch_mean(t)
                                # -> [B, C, T_tokens-1, H_tokens, W_tokens]
                                target_action = (
                                    patch_means[:, :, 1:]
                                    - patch_means[:, :, :-1]
                                )
                                # -> [B, T_tokens-1, H_tokens, W_tokens, C]
                                target_action = target_action.permute(
                                    0, 2, 3, 4, 1
                                )
                                if ac_target_dim != C_clip:
                                    target_action = target_action[
                                        ..., :ac_target_dim
                                    ]
                            for zi, mi in zip(z_seq, m_seq):
                                B, K, D = zi.shape
                                if mi.shape[1] > N:
                                    continue
                                # Reconstruct full grid for student z
                                full_z = torch.zeros(
                                    B, N, D, device=zi.device, dtype=zi.dtype
                                )
                                full_z.scatter_(
                                    1,
                                    mi.unsqueeze(-1).expand(-1, -1, D),
                                    zi,
                                )
                                full_z = full_z.view(
                                    B, T_tokens, H_tokens, W_tokens, D
                                )
                                # Predict action at time t from encoder feature
                                # at time t (excluding last timestep).
                                z_for_action = full_z[:, :-1]
                                
                                # FAC-JEPA: If FWM is active, only use Z_dyn
                                if fwm_static_coeff > 0.0 or fwm_orth_coeff > 0.0:
                                    D_app = max(1, int(D * fwm_app_ratio))
                                    z_for_action = z_for_action[..., D_app:]
                                    
                                predicted_action = action_head(z_for_action)
                                # Valid mask: both t and t+1 in context.
                                valid_mask = torch.zeros(
                                    B, N, device=zi.device, dtype=torch.bool
                                )
                                valid_mask.scatter_(1, mi, True)
                                valid_mask = valid_mask.view(
                                    B, T_tokens, H_tokens, W_tokens
                                )
                                valid_pairs = (
                                    valid_mask[:, :-1] & valid_mask[:, 1:]
                                )
                                if valid_pairs.any():
                                    tgt = target_action.to(
                                        predicted_action.dtype
                                    )
                                    action_err = torch.abs(
                                        predicted_action - tgt
                                    )
                                    ac_loss = (
                                        ac_loss
                                        + action_err[valid_pairs].mean()
                                    )
                                    n_ac += 1
                        if n_ac > 0:
                            ac_loss = ac_loss / n_ac
                            loss = loss + ac_coeff * ac_loss

                    # Kinematic-JEPA: Temporal Difference Sparsity Loss
                    if kinematic_coeff > 0.0:
                        import math
                        current_kinematic_coeff = kinematic_coeff
                        if kinematic_anneal:
                            # Cosine decay from kinematic_coeff to 0.0 over num_epochs
                            current_kinematic_coeff = kinematic_coeff * 0.5 * (1 + math.cos(math.pi * epoch / num_epochs))
                            
                        if current_kinematic_coeff > 0.0:
                            kinematic_loss = 0.0
                            for i, hi in enumerate(h):
                                B, L, D = hi.shape
                                D_kinematic = int(D * kinematic_split_ratio)
                                
                                T_tokens = clips[i].shape[2] // model_tubelet_size
                                H_tokens = clips[i].shape[3] // patch_size
                                W_tokens = clips[i].shape[4] // patch_size
                                
                                hi_spatial = hi[:, :, :D_kinematic]
                                if has_cls_first:
                                    hi_spatial = hi[:, 1:, :D_kinematic]
                                
                                if hi_spatial.shape[1] == T_tokens * H_tokens * W_tokens:
                                    hi_reshaped = hi_spatial.view(B, T_tokens, H_tokens, W_tokens, D_kinematic)
                                    
                                    if kinematic_type == "acceleration":
                                        if T_tokens >= 3:
                                            accel_z = hi_reshaped[:, 2:] - 2 * hi_reshaped[:, 1:-1] + hi_reshaped[:, :-2]
                                            kinematic_loss += torch.abs(accel_z).mean()
                                        else:
                                            delta_z = hi_reshaped[:, 1:] - hi_reshaped[:, :-1]
                                            kinematic_loss += torch.abs(delta_z).mean()
                                    elif kinematic_type == "huber":
                                        delta_z = hi_reshaped[:, 1:] - hi_reshaped[:, :-1]
                                        kinematic_loss += torch.nn.functional.huber_loss(delta_z, torch.zeros_like(delta_z), delta=0.1)
                                    else:
                                        delta_z = hi_reshaped[:, 1:] - hi_reshaped[:, :-1]
                                        kinematic_loss += torch.abs(delta_z).mean()
                                else:
                                    logger.warning(f"Kinematic-JEPA: L ({hi_spatial.shape[1]}) != T*H*W ({T_tokens*H_tokens*W_tokens})")
                            
                            kinematic_loss /= len(h)
                            loss += current_kinematic_coeff * kinematic_loss

                # Step 2. Backward & step
                run_step = True
                if loss_reg_std_mult is not None:
                    meanval = np.mean(trailing_losses)
                    stdval = np.std(trailing_losses)
                    max_bound = meanval + loss_reg_std_mult * stdval
                    if (
                        loss > max_bound
                        and epoch > loss_reg_min_epoch
                        and len(trailing_losses)
                        > int(0.5 * loss_reg_num_tracking_steps)
                    ):
                        run_step = False
                        loss.backward()
                        logger.info(
                            f"Loss {loss} is above bound {meanval} + {loss_reg_std_mult} * {stdval}. Skipping step."
                        )

                if run_step:
                    if mixed_precision:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()
                    if mixed_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                m = min(next(momentum_scheduler), ema[1])
                if not disable_ema:
                    with torch.no_grad():
                        params_k = []
                        params_q = []
                        for param_q, param_k in zip(
                            encoder.parameters(), target_encoder.parameters()
                        ):
                            params_k.append(param_k)
                            params_q.append(param_q)
                        torch._foreach_mul_(params_k, m)
                        torch._foreach_add_(params_k, params_q, alpha=1 - m)

                return (
                    float(loss),
                    _new_lr,
                    _new_wd,
                    run_step,
                )

            (
                loss,
                _new_lr,
                _new_wd,
                run_step,
            ), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            if loss_reg_std_mult is not None:
                if run_step:
                    trailing_losses.append(loss)
                    if len(trailing_losses) > loss_reg_num_tracking_steps:
                        trailing_losses = trailing_losses[1:]
                else:
                    step_count += 1
                    if step_count > MAX_REPEAT_COUNTS:
                        raise RuntimeError(
                            "Loss is above bound for too many tries. Exiting."
                        )

            # -- Logging
            def log_stats():
                csv_logger.log(
                    epoch + 1,
                    itr,
                    loss,
                    iter_elapsed_time_ms,
                    gpu_etime_ms,
                    data_elapsed_time_ms,
                )
                if (
                    (itr % log_freq == 0)
                    or (itr == ipe - 1)
                    or np.isnan(loss)
                    or np.isinf(loss)
                ):
                    logger.info(
                        "[%d, %5d] loss: %.3f "
                        "masks: %s "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] "
                        "[gpu: %.1f ms] "
                        "[data: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            "["
                            + ", ".join(
                                [
                                    f"{k}: " + "%.1f" % mask_meters[k].avg
                                    for k in mask_meters
                                ]
                            )
                            + "]",
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )

            log_stats()
            assert not np.isnan(loss), "loss is nan"

        # -- Save Checkpoint
        logger.info("avg. loss %.3f" % loss_meter.avg)
        if (epoch + 1) % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and (epoch + 1) % save_every_freq == 0:
                save_every_file = f"e{epoch}.pth.tar"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)
