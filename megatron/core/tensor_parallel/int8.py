import torch
# import torch_npu
from torch.nn.functional import linear as _torch_linear_forward_op
try:
    from torchao.prototype.quantized_training.int8_mm import scaled_int8_mm, scaled_int8_mm_cuda
except (ModuleNotFoundError, ImportError):
    print('torchao not found', flush=True)

col_scale_base = torch.tensor([1.0], dtype=torch.float32)
int8_pad = True


@torch.jit.script
def stochastic_round(x: torch.Tensor) -> torch.Tensor:
    floor_x = x.floor()                  # 得到整数部分 ⌊x⌋
    frac_x = x - floor_x                 # 得到小数部分 δ = x - ⌊x⌋
    # rand = torch.rand_like(x)           # [0, 1) 均匀分布
    rand = torch.empty_like(x, dtype=torch.float16, device=x.device).uniform_(0.0, 1.0)
    return floor_x + (rand < frac_x).to(x.dtype)


def quick_quantiles(x: torch.Tensor, qs: list):
    """
    Calculate multiple quantiles of a tensor using one sorting step.
    """
    if x is None or len(qs) == 0:
        return {}

    # Ensure the tensor is flattened for consistent quantile calculation
    n = x.numel()
    x_flat = x.flatten()

    # Sort the tensor once
    sorted_x, _ = torch.sort(x_flat)

    # Calculate the indices for each quantile
    indices = [int(q * n) for q in qs]

    # Extract the quantiles using the pre-sorted tensor
    # quantiles = {q: sorted_x[i].item() for q, i in zip(qs, indices)}
    quantiles = torch.tensor([sorted_x[i] for i in indices])

    return quantiles

def clamp_outliers(x: torch.Tensor, p: float = 0.999) -> tuple[torch.Tensor, torch.Tensor]:
    """
    筛选出分位数为p的outlier值和剔除过outlier的原tensor
    Args:
        x: 输入tensor
        p: 分位数, 将最大和最小的 p*numel个outlier值clamp, p = (0, 1)
    Returns:
        x_clean: 剔除过outlier的原tensor, outlier位置置0
        outliers: outlier值
    """
    # low_val = quick_quantile(x, 1 - p)
    # high_val = quick_quantile(x, p)
    quantiles = quick_quantiles(x, [1 - p, p]).tolist()
    outlier_mask = (x < quantiles[0]) | (x > quantiles[1])

    outliers = torch.where(outlier_mask, x, torch.zeros_like(x))
    x_clean = x.masked_fill(outlier_mask, 0)

    return x_clean, outliers


def per_row_quantize_int8(x: torch.Tensor, sr=True) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, f"{x.dim()}, {x.shape}"
    x_amax = x.abs().float().amax(dim=1).clamp(min=1e-4)
    scale = 127 / x_amax  # int8范围是 [-127,127]（保留0）
    if sr:
        x_scaled = stochastic_round((x * scale[:, None])).clamp(-127, 127).to(torch.int8)
    else:
        x_scaled = (x * scale[:, None]).round().clamp(-127, 127).to(torch.int8)

    return x_scaled, x_amax / 127


def per_col_quantize_int8(x: torch.Tensor, sr=True) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, f"{x.dim()}, {x.shape}"
    x_amax = x.abs().float().amax(dim=0).clamp(min=1e-4)
    scale = 127 / x_amax  # int8范围是 [-127,127]（保留0）
    if sr:
        x_scaled = stochastic_round((x * scale[None, :])).clamp(-127, 127).to(torch.int8)
    else:
        x_scaled = (x * scale[None, :]).round().clamp(-127, 127).to(torch.int8)

    return x_scaled, x_amax / 127

def per_token_scaled_int8_mm(
    A: torch.Tensor, B: torch.Tensor, row_scale: torch.Tensor, col_scale: torch.Tensor
) -> torch.Tensor:
    return torch._int_mm(A, B) * col_scale.view(-1) * row_scale.view(-1, 1)


# scale in float32
def per_tensor_quantize_int8(x: torch.Tensor, sr=True) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, f"{x.dim()}, {x.shape}"
    x_amax = x.abs().float().amax().clamp(min=1e-4)
    scale = 127 / x_amax
    if sr:
        x_scaled = stochastic_round(x * scale).clamp(-127, 127).to(torch.int8)
    else:
        x_scaled = (x * scale).round().clamp(-127, 127).to(torch.int8)

    return x_scaled, x_amax / 127

def per_tensor_scaled_int8_mm(
    A: torch.Tensor, B: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor
) -> torch.Tensor:
    return torch._int_mm(A, B) * a_scale * b_scale


def per_token_tensor_scaled_int8_mm(
    A: torch.Tensor, B: torch.Tensor, row_scale: torch.Tensor, b_scale: torch.Tensor
) -> torch.Tensor:
    return torch._int_mm(A, B)  * b_scale * row_scale.view(-1, 1)

def per_tensor_token_scaled_int8_mm(
    A: torch.Tensor, B: torch.Tensor, a_scale: torch.Tensor, col_scale: torch.Tensor
) -> torch.Tensor:
    return torch._int_mm(A, B)  * col_scale.view(-1) * a_scale


def padding_to_group_size(x: torch.Tensor, group_size: int, dim: int = -1, value: float = 0.0) -> tuple[torch.Tensor, int]:
    size = x.size(dim)
    remainder = size % group_size
    if remainder == 0:
        return x, 0
    pad_size = group_size - remainder
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_size
    pad_tensor = torch.full(pad_shape, value, dtype=x.dtype, device=x.device)
    x_padded = torch.cat([x, pad_tensor], dim=dim)
    return x_padded, pad_size


def per_group_row_quantize_int8(x: torch.Tensor, sr=True, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    M, K = x.shape
    if int8_pad:
        x_padded, pad_K = padding_to_group_size(x, group_size, dim=1)
    else:
        x_padded = x
    K_pad = x_padded.shape[1]
    assert x_padded.size(1) % group_size == 0, f"{x.shape}"

    num_groups = K_pad // group_size

    x_amax = x_padded.abs().float().reshape(M, num_groups, group_size).amax(dim=2).clamp(min=1e-4)
    scale = 127 / x_amax
    scale = scale.unsqueeze(2).expand(-1, -1, group_size).reshape(M, K_pad)

    if sr:
        x_scaled = stochastic_round((x_padded * scale)).clamp(-127, 127).to(torch.int8)
    else:
        x_scaled = (x_padded * scale).round().clamp(-127, 127).to(torch.int8)

    return x_scaled, x_amax / 127


def per_group_col_quantize_int8(x: torch.Tensor, sr=True, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    M, K = x.shape
    if int8_pad:
        x_padded, pad_M = padding_to_group_size(x, group_size, dim=0)
    else:
        x_padded = x
    M_pad = x_padded.shape[0]
    num_groups = M_pad // group_size

    x_amax = x_padded.abs().float().reshape(num_groups, group_size, K).amax(dim=1).clamp(min=1e-4)
    scale = 127 / x_amax
    scale = scale.unsqueeze(1).expand(-1, group_size, -1).reshape(M_pad, K)

    if sr:
        x_scaled = stochastic_round((x_padded * scale)).clamp(-127, 127).to(torch.int8)
    else:
        x_scaled = (x_padded * scale).round().clamp(-127, 127).to(torch.int8)
    
    return x_scaled, x_amax / 127


def per_block_quantize_int8(x: torch.Tensor, sr: bool = True, row_group_size: int = 128, col_group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    M, K = x.shape
    if int8_pad:
        x_padded, pad_K = padding_to_group_size(x, row_group_size, dim=1)
        x_padded, pad_M = padding_to_group_size(x_padded, col_group_size, dim=0)
    else:
        x_padded = x

    assert x_padded.size(0) % row_group_size == 0 and x_padded.size(1) % col_group_size == 0, f"{x.shape}"
    M_pad, K_pad = x_padded.shape

    num_row_groups = M_pad // row_group_size
    num_col_groups = K_pad // col_group_size

    x_blocks = x_padded.float().reshape(num_row_groups, row_group_size, num_col_groups, col_group_size)
    block_amax = x_blocks.abs().amax(dim=(1, 3)).clamp(min=1e-4)
    scale = 127.0 / block_amax

    scale_expanded = scale.unsqueeze(1).unsqueeze(3).expand(-1, row_group_size, -1, col_group_size).reshape(M_pad, K_pad)

    if sr:
        x_q = stochastic_round(x_padded * scale_expanded).clamp(-127, 127).to(torch.int8)
    else:
        x_q = (x_padded * scale_expanded).round().clamp(-127, 127).to(torch.int8)

    inv_scale = block_amax / 127.0
    return x_q, inv_scale


def per_group_scaled_int8_mm(
    A_q: torch.Tensor,
    B_q: torch.Tensor,
    a_inv_scale: torch.Tensor,
    b_inv_scale: torch.Tensor,
    group_size: int = 128
) -> torch.Tensor:
    # A_q: [M, K], B_q: [K, N]
    M, K = A_q.shape
    _, N = B_q.shape
    G = K // group_size
    # accumulate results
    out = torch.zeros((M, N), dtype=torch.float32, device=A_q.device)
    # perform per-group int8 matmul with scaling
    for g in range(G):
        i0, i1 = g * group_size, (g + 1) * group_size
        # partial product
        part = torch._int_mm(A_q[:, i0:i1], B_q[i0:i1, :])
        # apply inverse scales per group
        # a_inv_scale[g] is shape [M], b_inv_scale[g] is shape [N]
        scales = a_inv_scale[:, g].unsqueeze(1) * b_inv_scale[g, :].unsqueeze(0)
        out += part.to(torch.float32) * scales
    return out


def per_rowgroup_block_scaled_int8_mm(
    A_q: torch.Tensor,  # [M, K]
    B_q: torch.Tensor,  # [K, N]
    A_inv_scale: torch.Tensor,  # [M, K // group_size]
    B_inv_scale: torch.Tensor,  # [K // row_group_size, N // col_group_size]
    row_group_size: int = 128,
    col_group_size: int = 128,
    group_size: int = 128
) -> torch.Tensor:
    M, K = A_q.shape
    _, N = B_q.shape    
    out = torch.zeros((M, N), dtype=torch.float32, device=A_q.device)

    num_k_groups = K // group_size
    num_b_col_groups = N // col_group_size

    for g in range(num_k_groups):
        i0 = g * group_size
        i1 = (g + 1) * group_size
        A_part = A_q[:, i0:i1]  # [M, group_size]
        B_part = B_q[i0:i1, :]  # [group_size, N]
        partial = torch._int_mm(A_part, B_part)  # [M, N]
        # A_inv_scale: [M, num_k_groups] → [M, 1]
        a_scale = A_inv_scale[:, g].unsqueeze(1)  # [M, 1]

        for c in range(num_b_col_groups):
            j0 = c * col_group_size
            j1 = (c + 1) * col_group_size
            # y 的行 block 是第 g 个（由 i0 决定）
            b_scale = B_inv_scale[g, c].unsqueeze(0)  # [1,]
            scale = a_scale * b_scale  # [M, 1] * [1,] = broadcast 到 [M, col_group_size]
            out[:, j0:j1] += partial[:, j0:j1].float() * scale

    return out

def per_block_scaled_int8_mm(
    A_q: torch.Tensor,  # [M, K]
    B_q: torch.Tensor,  # [K, N]
    A_inv_scales: torch.Tensor,  # [M // R, K // C]
    B_inv_scales: torch.Tensor,  # [K // R, N // C]
    row_group_size: int = 128,
    col_group_size: int = 128
) -> torch.Tensor:
    M, K = A_q.shape
    _, N = B_q.shape

    out = torch.zeros((M, N), dtype=torch.float32, device=A_q.device)

    num_row_blocks = M // row_group_size
    num_col_blocks = N // col_group_size
    num_k_blocks = K // col_group_size

    for r in range(num_row_blocks):
        i0 = r * row_group_size
        i1 = i0 + row_group_size, M
        for c in range(num_col_blocks):
            j0 = c * col_group_size
            j1 = j0 + col_group_size
            out_block = torch.zeros((row_group_size, col_group_size), dtype=torch.float32, device=A_q.device)
            for k in range(num_k_blocks):
                k0 = k * col_group_size
                k1 = k0 + col_group_size
                A_block = A_q[i0:i1, k0:k1]
                B_block = B_q[k0:k1, j0:j1]
                # int8 matmul
                partial = torch._int_mm(A_block, B_block)  # [row_group_size, col_group_size]
                a_scale = A_inv_scales[r, k]
                b_scale = B_inv_scales[k, c]
                scale = a_scale * b_scale
                out_block += partial.float() * scale

            out[i0:i1, j0:j1] = out_block
    return out







def _select_int8_ops(mode):
    # assert mode == "tile_block"
    if mode == "channel": # token * token
        return per_row_quantize_int8, per_col_quantize_int8, scaled_int8_mm
    elif mode == "tensor": # tensor * tensor
        return per_tensor_quantize_int8, per_tensor_quantize_int8, per_tensor_scaled_int8_mm
    elif mode == "tile": # tile * tile, tile = 1*128
        return per_group_row_quantize_int8, per_group_col_quantize_int8, per_group_scaled_int8_mm
    elif mode == "tile_block": # tile * block, tile = 1*128 block = 128*128
        return per_group_row_quantize_int8, per_block_quantize_int8, per_rowgroup_block_scaled_int8_mm
    elif mode == "block": # block * blcok
        return per_block_quantize_int8, per_block_quantize_int8, per_block_scaled_int8_mm
    elif mode == "channel_tensor": # token * tensor
        return per_row_quantize_int8, per_tensor_quantize_int8, per_token_tensor_scaled_int8_mm
    elif mode == "tensor_channel":
        return per_tensor_quantize_int8, per_col_quantize_int8, per_tensor_token_scaled_int8_mm
    else:
        assert False


# @dump_quantize_wrapper
def _quantize_int8(A, B, mode="channel", sr=False, trans_b=True):
    quantize_A, quantize_B, _ = _select_int8_ops(mode)

    A_int8, A_scale = quantize_A(A, sr=sr)
    if trans_b:
        B_int8, B_scale = quantize_B(B.t(), sr=sr)
    else:
        B_int8, B_scale = quantize_B(B, sr=sr)
    return A_int8, B_int8, A_scale, B_scale