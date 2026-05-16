import torch
import torch.nn as nn

class HamiltonianNN(nn.Module):
    """
    A simple MLP that takes [q, p] and outputs a scalar Energy H(q, p).
    """
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        # Input is q and p concatenated, so 2 * dim
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1) # Outputs scalar Energy
        )
        
    def forward(self, q, p):
        # q, p shape: [B, Tokens, D/2]
        x = torch.cat([q, p], dim=-1) # [B, Tokens, D]
        H = self.net(x) # [B, Tokens, 1]
        return H.sum() # Total energy of the system

def symplectic_euler_step(q, p, H_net, dt=0.1):
    """
    Performs one step of Symplectic Euler integration.
    q_new = q_old + dt * (dH/dp)
    p_new = p_old - dt * (dH/dq_new)
    """
    # We need gradients with respect to q and p
    q.requires_grad_(True)
    p.requires_grad_(True)
    
    # 1. Calculate H(q, p)
    H_val = H_net(q, p)
    
    # 2. Calculate dH/dp to update q
    dH_dp = torch.autograd.grad(H_val, p, create_graph=True)[0]
    q_new = q + dt * dH_dp
    
    # 3. Calculate H(q_new, p) to update p
    H_val_new = H_net(q_new, p)
    dH_dq_new = torch.autograd.grad(H_val_new, q_new, create_graph=True)[0]
    p_new = p - dt * dH_dq_new
    
    return q_new, p_new
