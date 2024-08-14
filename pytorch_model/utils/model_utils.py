import torch
from typing import Tuple

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float16)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def precompute_freqs_cis_rect(dim:int, end:int, theta:float = 10000.0):
    return torch.view_as_real(precompute_freqs_cis(dim, end, theta))

def precompute_freqs_cis_rect_exp(dim: int, end: int, theta: float = 10000.0):
    # TODO: VERIFY CORRECTNESS.
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float16)
    freqs = torch.outer(t, freqs)
    freqs_real = torch.cos(freqs)
    freqs_imag = torch.sin(freqs)
    freqs_cis_rect = torch.stack((freqs_real, freqs_imag), dim=-1)
    return freqs_cis_rect

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    # assert 0 <= 1 < ndim
    # assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb_rectangular(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    xq = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk = xk.float().reshape(*xk.shape[:-1], -1, 2)

    xq_real = xq[..., 0]
    xq_imag = xq[..., 1]
    
    xk_real = xk[..., 0]
    xk_imag = xk[..., 1]
    
    freqs_cis_real = reshape_for_broadcast(freqs_cis[..., 0], xq_real)
    freqs_cis_imag = reshape_for_broadcast(freqs_cis[..., 1], xq_imag)
    
    # Apply rotary embedding
    xq_real_out = xq_real * freqs_cis_real - xq_imag * freqs_cis_imag
    xq_imag_out = xq_real * freqs_cis_imag + xq_imag * freqs_cis_real
    
    xk_real_out = xk_real * freqs_cis_real - xk_imag * freqs_cis_imag
    xk_imag_out = xk_real * freqs_cis_imag + xk_imag * freqs_cis_real
    
    # Stack the real and imaginary parts together
    xq_out = torch.stack((xq_real_out, xq_imag_out), dim=-1)
    xk_out = torch.stack((xk_real_out, xk_imag_out), dim=-1)
    
    # Flatten the output tensors
    xq_out = xq_out.flatten(-2)
    xk_out = xk_out.flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)