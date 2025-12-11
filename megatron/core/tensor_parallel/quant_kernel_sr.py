# 测试triton 2025-10-8
# 首先实现简单的tensor量化和GEMM kernel

import math
import triton
import triton.language as tl
import torch

int8_pad = True

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

###################################################col 量化实现################################################
# @triton.jit
# def per_col_quantize_int8_kernel(
#     input_ptr,       # fp16/fp32 原始矩阵  (M, N)
#     output_ptr,      # int8 量化结果       (M, N)  行优先
#     col_max_ptr,     # 每列绝对最大值      (N,)
#     M: tl.constexpr,
#     N: tl.constexpr,
#     NP2: tl.constexpr,
#     sr: tl.constexpr,
# ):
#     pid = tl.program_id(0)          # 一列一个 program
#     arange = tl.arange(0, NP2)      # 0..M-1 (补齐到 2 幂)
#     mask = arange < M

#     # 行优先索引:  offset = row * N + col
#     offsets = arange * N + pid
#     x = tl.load(input_ptr + offsets, mask=mask)        # 整列数据

#     # x_abs = tl.abs(x)
#     # col_max = tl.max(tl.where(mask, x_abs, 0), axis=0)  # 整列最大值

#     # scale = 127.0 / col_max
#     # scaled = x * scale                                    # 127 * (x / max)

#     x_f32 = x.to(tl.float32)
#     x_abs = tl.abs(x_f32)
#     col_max = tl.max(tl.where(mask, x_abs, float("-inf")), axis=0)
    
#     col_max_safe  = tl.maximum(col_max, 1e-4)


#     scale = 127.0 / col_max_safe 
#     scaled = x_f32 * scale
#     if sr:        # stochastic rounding
#         floor_val = tl.floor(scaled)
#         frac = scaled - floor_val
#         seed = 0
#         offset = pid * NP2 + arange
#         rand_val = tl.rand(seed, offset)
#         rounded = floor_val + (rand_val < frac).to(scaled.dtype)
#     else:         # round-to-nearest ( ties to even )
#         rounded = tl.extra.cuda.libdevice.round(scaled)
#     tl.store(output_ptr + offsets, rounded.to(tl.int8), mask=mask)
#     tl.store(col_max_ptr + pid, col_max_safe)


# def per_col_quantize_int8_triton(x: torch.Tensor, sr: bool = False):
#     assert x.is_cuda, "input must be CUDA tensor"
#     x = x.contiguous()
#     M, N = x.shape
#     output = torch.empty(M, N, dtype=torch.int8, device=x.device)   # 保持相同形状
#     col_maxs = torch.empty(N, dtype=torch.float32, device=x.device)  # 每列一个标量

#     NP2 = triton.next_power_of_2(M)
#     grid = lambda meta: (N,)

#     per_col_quantize_int8_kernel[grid](
#         x,
#         output,
#         col_maxs,
#         M=M,
#         N=N,
#         NP2=NP2,
#         sr=sr,
#     )
#     return output, col_maxs


# ###################################################row 量化实现################################################

# @triton.jit
# def per_row_quantize_int8_kernel(
#         input_ptr,      # fp16/fp32 输入 (M, N)
#         output_ptr,     # int8 输出 (M, N)
#         row_max_ptr,    # fp16 每行最大值 (M,)
#         n_elements,     # 仅用于 autotune
#         M: tl.constexpr,
#         N: tl.constexpr,
#         BLOCK_SIZE: tl.constexpr,
#         NP2: tl.constexpr,
#         sr: tl.constexpr):  # 新增：是否使用stochastic rounding
#     pid = tl.program_id(0)          # 行号

#     cols = tl.arange(0, NP2)
#     offs = pid * N + cols           # 线性偏移
#     mask = cols < N

#     row = tl.load(input_ptr + offs, mask=mask)
#     row_f32 = row.to(tl.float32)
#     row_abs = tl.abs(row_f32)
#     row_max = tl.max(tl.where(mask, row_abs, float("-inf")), axis=0)
#     row_max_safe = tl.maximum(row_max, 1e-4)  # 对齐 Python 的 clamp

#     scale = 127.0 / row_max_safe
#     scaled_row = row_f32 * scale
    
#     # 根据sr选择舍入方式
#     if sr:
#         # stochastic rounding实现
#         floor_scaled = tl.floor(scaled_row)
#         frac = scaled_row - floor_scaled
        
#         # 生成随机数 (使用当前行号和列位置作为随机种子的一部分)
#         seed = 0  # 可以使用固定种子
#         offset = pid * NP2 + cols
#         rand_val = tl.rand(seed, offset)
        
#         # stochastic rounding: 以概率frac向上取整
#         rounded = floor_scaled + (rand_val < frac).to(scaled_row.dtype)
#     else:
#         # 原来的四舍五入
#         rounded = tl.extra.cuda.libdevice.round(scaled_row)

#     q_int8 = rounded.to(tl.int8)

#     tl.store(output_ptr + offs, q_int8, mask=mask)
#     tl.store(row_max_ptr + pid, row_max_safe)


# def per_row_quantize_int8_triton(x: torch.Tensor, sr: bool = False):
#     assert x.is_cuda
#     x = x.contiguous()
#     M, N = x.shape
#     out = torch.empty_like(x, dtype=torch.int8)
#     # row_max = torch.empty(M, dtype=torch.float16, device=x.device)
#     row_max = torch.empty(M, dtype=torch.float32, device=x.device)

#     NP2 = triton.next_power_of_2(N)
#     grid = lambda meta: (M,)

#     per_row_quantize_int8_kernel[grid](
#         x, out, row_max,
#         n_elements=x.numel(),
#         M=M, N=N,
#         BLOCK_SIZE=N,
#         NP2=NP2,
#         sr=sr)  # 传入sr参数
#     return out, row_max


###################################################col 量化实现################################################
@triton.jit
def per_col_quantize_int8_kernel(
    input_ptr, output_ptr, col_max_ptr,
    M: tl.constexpr, N: tl.constexpr, NP2: tl.constexpr, sr: tl.constexpr,
):
    pid = tl.program_id(0)
    arange = tl.arange(0, NP2)
    mask = arange < M

    offsets = arange * N + pid
    x = tl.load(input_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    x_abs = tl.abs(x_f32)
    
    # 计算最大值（使用更小的clamp阈值）
    col_max = tl.max(tl.where(mask, x_abs, float("-inf")), axis=0)
    col_max_safe = tl.maximum(col_max, 1e-7)  # 降低阈值
    
    # 计算缩放值
    abs_max_inv = 1.0 / col_max_safe
    scaled = 127.0 * (x_f32 * abs_max_inv)
    
    # 舍入处理
    if sr:
        floor_val = tl.floor(scaled)
        frac = scaled - floor_val
        seed = 0
        offset = pid * NP2 + arange
        rand_val = tl.rand(seed, offset)
        rounded = floor_val + (rand_val < frac).to(scaled.dtype)
    else:
        rounded = tl.extra.cuda.libdevice.round(scaled)
    
    # 使用tl.clamp进行边界限制（最简洁高效）
    clamped = tl.clamp(rounded, -127.0, 127.0)
    
    # 存储结果
    tl.store(output_ptr + offsets, clamped.to(tl.int8), mask=mask)
    tl.store(col_max_ptr + pid, col_max_safe)

def per_col_quantize_int8_triton(x: torch.Tensor, sr: bool = False):
    assert x.is_cuda, "input must be CUDA tensor"
    x = x.contiguous()
    M, N = x.shape
    output = torch.empty(M, N, dtype=torch.int8, device=x.device)   # 保持相同形状
    col_maxs = torch.empty(N, dtype=torch.float32, device=x.device)  # 每列一个标量

    NP2 = triton.next_power_of_2(M)
    grid = lambda meta: (N,)

    per_col_quantize_int8_kernel[grid](
        x,
        output,
        col_maxs,
        M=M,
        N=N,
        NP2=NP2,
        sr=sr,
    )
    return output, col_maxs


###################################################row 量化实现################################################

@triton.jit
def per_row_quantize_int8_kernel(
        input_ptr, output_ptr, row_max_ptr,
        n_elements, M: tl.constexpr, N: tl.constexpr, 
        BLOCK_SIZE: tl.constexpr, NP2: tl.constexpr, sr: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, NP2)
    offs = pid * N + cols
    mask = cols < N

    row = tl.load(input_ptr + offs, mask=mask)
    row_f32 = row.to(tl.float32)
    row_abs = tl.abs(row_f32)
    
    row_max = tl.max(tl.where(mask, row_abs, float("-inf")), axis=0)
    row_max_safe = tl.maximum(row_max, 1e-7)
    
    abs_max_inv = 1.0 / row_max_safe
    scaled_row = 127.0 * (row_f32 * abs_max_inv)
    
    if sr:
        floor_scaled = tl.floor(scaled_row)
        frac = scaled_row - floor_scaled
        seed = 0
        offset = pid * NP2 + cols
        rand_val = tl.rand(seed, offset)
        rounded = floor_scaled + (rand_val < frac).to(scaled_row.dtype)
    else:
        rounded = tl.extra.cuda.libdevice.round(scaled_row)
    
    # 使用tl.clamp
    clamped = tl.clamp(rounded, -127.0, 127.0)
    
    tl.store(output_ptr + offs, clamped.to(tl.int8), mask=mask)
    tl.store(row_max_ptr + pid, row_max_safe)

def per_row_quantize_int8_triton(x: torch.Tensor, sr: bool = False):
    assert x.is_cuda
    x = x.contiguous()
    M, N = x.shape
    out = torch.empty_like(x, dtype=torch.int8)
    # row_max = torch.empty(M, dtype=torch.float16, device=x.device)
    row_max = torch.empty(M, dtype=torch.float32, device=x.device)

    NP2 = triton.next_power_of_2(N)
    grid = lambda meta: (M,)

    per_row_quantize_int8_kernel[grid](
        x, out, row_max,
        n_elements=x.numel(),
        M=M, N=N,
        BLOCK_SIZE=N,
        NP2=NP2,
        sr=sr)  # 传入sr参数
    return out, row_max

###################################################tensor 量化实现################################################
# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
#         triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
#         triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
#     ],
#     key=['n_elements'],
# )
# @triton.jit
# def per_tensor_quantize_int8_kernel(
#     x_ptr, # Input
#     abs_max_inv_ptr, # Pointer to scaling factor 1/abs(max_value)
#     y_ptr, # Output
#     n_elements, # Number of elements in input
#     BLOCK_SIZE: tl.constexpr,
#     sr: tl.constexpr):  # 新增：是否使用stochastic rounding

#     pid = tl.program_id(0)
#     offsets = pid * BLOCK_SIZE  + tl.arange(0, BLOCK_SIZE)
#     mask = offsets < n_elements
#     x = tl.load(x_ptr + offsets, mask=mask)
#     abs_max_inv =  tl.load(abs_max_inv_ptr)

#     # 计算缩放后的值
#     scaled = 127.0 * (x * abs_max_inv)
#     # 根据sr选择舍入方式
#     if sr:
#         # stochastic rounding实现
#         floor_scaled = tl.floor(scaled)
#         frac = scaled - floor_scaled
        
#         # 生成随机数 (使用全局偏移作为随机种子的一部分)
#         seed = 0  # 可以使用固定种子
#         rand_val = tl.rand(seed, offsets)
        
#         # stochastic rounding: 以概率frac向上取整
#         y = floor_scaled + (rand_val < frac).to(scaled.dtype)
#     else:
#         # 原来的四舍五入
#         y = tl.extra.cuda.libdevice.round(scaled)
        
#     tl.store(y_ptr + offsets, y.to(tl.int8), mask = mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def per_tensor_quantize_int8_kernel(
    x_ptr, abs_max_inv_ptr, y_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr, sr: tl.constexpr):

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    abs_max_inv = tl.load(abs_max_inv_ptr)

    scaled = 127.0 * (x * abs_max_inv)
    
    if sr:
        floor_scaled = tl.floor(scaled)
        frac = scaled - floor_scaled
        seed = 0
        rand_val = tl.rand(seed, offsets)
        y = floor_scaled + (rand_val < frac).to(scaled.dtype)
    else:
        y = tl.extra.cuda.libdevice.round(scaled)
    
    # 使用tl.clamp
    clamped = tl.clamp(y, -127.0, 127.0)
    
    tl.store(y_ptr + offsets, clamped.to(tl.int8), mask=mask)

def per_tensor_quantize_int8_triton(x: torch.Tensor, sr: bool =False):
# def per_tensor_quantize_int8_triton(x: torch.Tensor, sr: bool = False, trans_b: bool =False):
    x = x.contiguous()
    abs_max = x.abs().max().unsqueeze(0).to(torch.float32) # [1,1]
    # abs_max = x.abs().float().amax().clamp(min=1e-4)
    # abs_max_inv = (1.0 / abs_max).to(torch.float16)
    abs_max_inv = (1.0 / abs_max)#.to(torch.float32)

    y = torch.empty(*x.shape, device = "cuda", dtype = torch.int8)

    assert y.is_cuda and x.is_cuda

    n_elements = x.numel()

    grid = lambda meta:(triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    per_tensor_quantize_int8_kernel[grid](
        x, abs_max_inv, y, n_elements, sr=sr  # 传入sr参数
    )

    return y, abs_max


#tile粒度版本##
#######################################per_group_row_quantize_int8_triton#######################

@triton.jit
def per_group_row_quantize_int8_kernel(
        input_ptr,      # fp16/fp32 输入 (M, N)
        output_ptr,     # int8 输出 (M, N)
        group_max_ptr,  # fp32 每组最大值 (M, num_groups)  # 修改：统一使用float32
        n_elements,     # 仅用于 autotune
        M: tl.constexpr,
        N: tl.constexpr,
        GROUP_SIZE: tl.constexpr,   # 每组列数
        NUM_GROUPS: tl.constexpr,
        NP2: tl.constexpr,
        sr: tl.constexpr):  # 新增：是否使用stochastic rounding
    pid_m = tl.program_id(0)          # 行号
    pid_g = tl.program_id(1)          # 组号

    cols = tl.arange(0, NP2) + pid_g * GROUP_SIZE
    mask = cols < N  # 修复：正确的边界检查，确保不越界
    offs = pid_m * N + cols
    row = tl.load(input_ptr + offs, mask=mask, other=0.0)
    
    # 计算绝对值并确保数值稳定性
    row_abs = tl.abs(row)
    # 添加最小值钳位，防止除零错误和数值不稳定
    group_max = tl.max(tl.where(mask, row_abs, 0.0), axis=0)
    group_max = tl.maximum(group_max, 1e-7)  # 恢复最小值钳位，确保数值稳定性

    # 安全的scale计算
    scale = 127.0 / group_max
    scaled_row = row * scale
    
    # 根据sr选择舍入方式
    if sr:
        # stochastic rounding实现
        floor_scaled = tl.floor(scaled_row)
        frac = scaled_row - floor_scaled
        
        # 生成随机数 (使用行号、组号和列位置作为随机种子的一部分)
        seed = 0  # 可以使用固定种子
        offset = pid_m * NUM_GROUPS * NP2 + pid_g * NP2 + cols
        rand_val = tl.rand(seed, offset)
        
        # stochastic rounding: 以概率frac向上取整
        rounded = floor_scaled + (rand_val < frac).to(scaled_row.dtype)
    else:
        # 原来的四舍五入
        rounded = tl.extra.cuda.libdevice.round(scaled_row)
    
    clamped = tl.clamp(rounded, -127.0, 127.0)
    q_int8 = clamped.to(tl.int8)

    tl.store(output_ptr + offs, q_int8, mask=mask)
    # 统一使用float32存储scale值，确保精度一致性
    tl.store(group_max_ptr + pid_m * NUM_GROUPS + pid_g, group_max.to(tl.float32))


def per_group_row_quantize_int8_triton(x: torch.Tensor, group_size: int = 128, sr: bool = False):
    assert x.is_cuda
    x = x.contiguous()
    M, N = x.shape
    assert N % group_size == 0, f"Input size {N} must be divisible by group_size {group_size}"
    num_groups = N // group_size
    out = torch.empty_like(x, dtype=torch.int8)
    # 统一使用float32数据类型，确保精度一致性
    group_max = torch.empty((M, num_groups), dtype=torch.float32, device=x.device)

    NP2 = triton.next_power_of_2(group_size)
    grid = (M, num_groups)

    per_group_row_quantize_int8_kernel[grid](
        x, out, group_max,
        n_elements=x.numel(),
        M=M, N=N,
        GROUP_SIZE=group_size,
        NUM_GROUPS=num_groups,
        NP2=NP2,
        sr=sr)  # 传入sr参数
    return out, group_max



#######################################per_group_col_quantize_int8_triton#######################
@triton.jit
def per_group_col_quantize_int8_kernel(
        input_ptr,
        output_ptr,
        group_max_ptr,
        n_elements,
        M: tl.constexpr,
        N: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        NP2: tl.constexpr,
        sr: tl.constexpr):  # 新增：是否使用stochastic rounding
    pid_n = tl.program_id(0)          # 列号
    pid_g = tl.program_id(1)          # 组号

    rows = tl.arange(0, NP2) + pid_g * GROUP_SIZE
    mask = rows < M  # 正确的边界检查
    offs = rows * N + pid_n
    col = tl.load(input_ptr + offs, mask=mask, other=0.0)
    
    # 计算绝对值并确保数值稳定性
    col_abs = tl.abs(col)
    group_max = tl.max(tl.where(mask, col_abs, 0.0), axis=0)
    group_max = tl.maximum(group_max, 1e-7)  # 确保最小值钳位
    
    # 安全的scale计算
    scale = 127.0 / group_max
    scaled = col * scale
    
    # 根据sr选择舍入方式
    if sr:
        # stochastic rounding实现
        floor_scaled = tl.floor(scaled)
        frac = scaled - floor_scaled
        
        # 生成随机数 (使用列号、组号和行位置作为随机种子的一部分)
        seed = 0  # 可以使用固定种子
        offset = pid_n * NUM_GROUPS * NP2 + pid_g * NP2 + rows
        rand_val = tl.rand(seed, offset)
        
        # stochastic rounding: 以概率frac向上取整
        q = floor_scaled + (rand_val < frac).to(scaled.dtype)
    else:
        # 使用Triton内置的round函数
        q = tl.extra.cuda.libdevice.round(scaled)
    
    clamped = tl.clamp(q, -127.0, 127.0)
    q_int8 = clamped.to(tl.int8)

    tl.store(output_ptr + offs, q_int8, mask=mask)
    tl.store(group_max_ptr + pid_g * N + pid_n, group_max.to(tl.float32))


def per_group_col_quantize_int8_triton(x: torch.Tensor, group_size: int = 128, sr: bool = False):
    assert x.is_cuda
    x = x.contiguous()
    M, N = x.shape
    num_groups = (M + group_size - 1) // group_size
    out = torch.empty((M, N), dtype=torch.int8, device=x.device)
    # 保持float32数据类型一致性
    group_max = torch.empty((num_groups, N), dtype=torch.float32, device=x.device)

    NP2 = triton.next_power_of_2(group_size)
    grid = (N, num_groups)

    per_group_col_quantize_int8_kernel[grid](
        x, out, group_max,
        n_elements=x.numel(),
        M=M, N=N,
        GROUP_SIZE=group_size,
        NUM_GROUPS=num_groups,
        NP2=NP2,
        sr=sr)  # 传入sr参数

    return out, group_max


#######################################per_block_quantize_int8_triton#######################

@triton.jit
def per_block_quantize_int8_kernel(
    x_ptr,           # fp16/fp32 输入 (M, N)
    y_ptr,           # int8 输出 (M, N)
    block_max_ptr,   # fp32 每 tile 最大值 (PADDED_M, PADDED_N)  # 修改：统一使用float32
    M: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    PADDED_M: tl.constexpr,
    PADDED_N: tl.constexpr,
    sr: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    offs = rm[:, None] * N + rn[None, :]

    tile = tl.load(x_ptr + offs, mask=mask)
    tile_abs = tl.abs(tile)
    tile_max = tl.max(tile_abs)
    
    # 修复：确保数值稳定性，使用更大的最小值钳位
    tile_max_safe = tl.maximum(tile_max, 1e-7)
    scale = 127.0 / tile_max_safe
    
    scaled_tile = tile * scale
    
    if sr:
        floor_scaled = tl.floor(scaled_tile)
        frac = scaled_tile - floor_scaled
        seed = pid_m * PADDED_N + pid_n
        element_idx = tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
        rand_val = tl.rand(seed, element_idx)
        q = floor_scaled + (rand_val < frac).to(scaled_tile.dtype)
    else:
        q = tl.extra.cuda.libdevice.round(scaled_tile)
    
    clamped = tl.clamp(q, -127.0, 127.0)
    q_int8 = clamped.to(tl.int8)

    tl.store(y_ptr + offs, q_int8, mask=mask)
    # 修复：存储原始的最大值，而不是缩放后的值
    tl.store(block_max_ptr + pid_m * PADDED_N + pid_n, tile_max_safe.to(tl.float32))


def per_block_quantize_int8_triton(x: torch.Tensor, tile: tuple = (128, 128), sr: bool = False):
    assert x.is_cuda
    x = x.contiguous()
    M, N = x.shape
    BM, BN = tile
    PADDED_M = (M + BM - 1) // BM
    PADDED_N = (N + BN - 1) // BN

    y = torch.empty_like(x, dtype=torch.int8)
    # 修复：统一使用float32存储最大值
    block_max = torch.empty((PADDED_M, PADDED_N), dtype=torch.float32, device=x.device)

    grid = (PADDED_M, PADDED_N)
    per_block_quantize_int8_kernel[grid](
        x, y, block_max,
        M, N, BM, BN, PADDED_M, PADDED_N, sr
    )
    return y, block_max




@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def per_tensor_scaled_int8_mm_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Scaling factors
    a_scale_ptr, b_scale_ptr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Kernel for computing the matrix multiplication of two int8 matrices with scaling.
    C = (A @ B) * a_scale * b_scale
    """
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Create block pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Load scaling factors
    a_scale = tl.load(a_scale_ptr).to(tl.float32)
    b_scale = tl.load(b_scale_ptr).to(tl.float32)
    total_scale = a_scale * b_scale
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load slices of A and B
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Compute matrix multiplication
        accumulator += tl.dot(a, b)
        
        # Update pointers for next iteration
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Apply scaling and convert to output type
    c = accumulator.to(tl.float32) * total_scale
    
    # Create output block pointers
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # Create mask for boundary checks
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    # Store result
    tl.store(c_ptrs, c, mask=mask)

def per_tensor_scaled_int8_mm_triton(
    A: torch.Tensor, B: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor
) -> torch.Tensor:
    """
    Matrix multiplication of two int8 matrices with scaling.
    
    Args:
        A: int8 tensor of shape (M, K)
        B: int8 tensor of shape (K, N)  
        a_scale: scale factor for A
        b_scale: scale factor for B
        
    Returns:
        float32 tensor of shape (M, N)
    """
    # Check inputs
    assert A.dtype == torch.int8, "A must be int8"
    assert B.dtype == torch.int8, "B must be int8"
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Matrix dimensions don't match: A[{M}, {K}] @ B[{K2}, {N}]"
    
    # Allocate output
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    # Ensure scales are on the same device and have correct shape
    a_scale = a_scale.to(A.device).reshape(1)
    b_scale = b_scale.to(A.device).reshape(1)
    
    # Grid launch
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    # Launch kernel
    per_tensor_scaled_int8_mm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), 
        C.stride(0), C.stride(1),
        a_scale, b_scale,
    )
    
    return C



##############################################per_channel_scaled_int8_mm_triton########################################

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def per_channel_scaled_int8_mm_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Scaling factors
    a_scale_ptr, b_scale_ptr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Kernel for computing the matrix multiplication of two int8 matrices with per-channel scaling.
    A is row-wise quantized, B is column-wise quantized.
    C = (A @ B) * (a_scale[i] * b_scale[j]) / (127 * 127)
    """
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Create block pointers for A and B
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Create pointers for scaling factors
    # A scaling factors are per-row (size M), B scaling factors are per-column (size N)
    a_scale_ptrs = a_scale_ptr + offs_am
    b_scale_ptrs = b_scale_ptr + offs_bn
    
    # Load scaling factors for this block
    a_scale_block = tl.load(a_scale_ptrs, mask=offs_am < M, other=0.0).to(tl.float32)
    b_scale_block = tl.load(b_scale_ptrs, mask=offs_bn < N, other=0.0).to(tl.float32)
    
    # Compute outer product of scaling factors for this block
    scale_matrix = a_scale_block[:, None] * b_scale_block[None, :]
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load slices of A and B
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N), other=0.0)
        
        # Compute matrix multiplication
        accumulator += tl.dot(a, b)
        
        # Update pointers for next iteration
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Apply scaling and convert to output type
    # Note: We divide by 127*127 because the original quantization used 127 as scaling factor
    c = (accumulator.to(tl.float32) * scale_matrix) # / (127.0 * 127.0)
    
    # Create output block pointers
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # Create mask for boundary checks
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    # Store result
    tl.store(c_ptrs, c, mask=mask)

def per_channel_scaled_int8_mm_triton(
    A: torch.Tensor, B: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor
) -> torch.Tensor:
    """
    Matrix multiplication of two int8 matrices with per-channel scaling.
    
    Args:
        A: int8 tensor of shape (M, K) - row-wise quantized
        B: int8 tensor of shape (K, N) - column-wise quantized  
        a_scale: scale factors for A, shape (M,) - per row
        b_scale: scale factors for B, shape (N,) - per column
        
    Returns:
        float32 tensor of shape (M, N)
    """
    # Check inputs
    assert A.dtype == torch.int8, "A must be int8"
    assert B.dtype == torch.int8, "B must be int8"
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Matrix dimensions don't match: A[{M}, {K}] @ B[{K2}, {N}]"
    
    # Check scaling factor shapes
    assert a_scale.shape == (M,), f"a_scale should have shape ({M},), got {a_scale.shape}"
    assert b_scale.shape == (N,), f"b_scale should have shape ({N},), got {b_scale.shape}"
    
    # Allocate output
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    # Ensure scales are on the same device
    a_scale = a_scale.to(A.device)
    b_scale = b_scale.to(A.device)
    
    # Grid launch
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    # Launch kernel
    per_channel_scaled_int8_mm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), 
        C.stride(0), C.stride(1),
        a_scale, b_scale,
    )
    
    return C


##############################################per_channel_scaled_int8_mm_triton########################################
#原始能收敛
# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_warps=4),
#         triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_warps=4),
#         triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_warps=4),
#         triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_warps=4),
#     ],
#     key=["M", "N", "K"],
# )
# @triton.jit
# def per_channel_scaled_int8_mm_kernel(
#     a_ptr, b_ptr, c_ptr,
#     M, N, K,
#     stride_am, stride_ak,
#     stride_bk, stride_bn,
#     stride_cm, stride_cn,
#     a_scale_ptr, b_scale_ptr,  # 现在传入的是原始的最大值，不是缩放因子
#     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
# ):
#     """
#     修复后的kernel：使用正确的反量化公式
#     C = (A_int8 @ B_int8) * (a_max[i] * b_max[j]) / (127.0 * 127.0)
#     """
#     pid = tl.program_id(axis=0)
#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
#     pid_m = pid // num_pid_n
#     pid_n = pid % num_pid_n
    
#     offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     offs_k = tl.arange(0, BLOCK_SIZE_K)
    
#     a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#     b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
#     # 加载原始的最大值
#     a_scale_ptrs = a_scale_ptr + offs_am
#     b_scale_ptrs = b_scale_ptr + offs_bn
    
#     a_max_block = tl.load(a_scale_ptrs, mask=offs_am < M, other=1.0).to(tl.float32)
#     b_max_block = tl.load(b_scale_ptrs, mask=offs_bn < N, other=1.0).to(tl.float32)
    
#     # 修复关键点：使用正确的反量化公式
#     # 量化公式：q = round(x * 127.0 / max_value)
#     # 反量化公式：x ≈ q * max_value / 127.0
#     # 所以矩阵乘法的缩放应该是 (a_max[i] * b_max[j]) / (127.0 * 127.0)
#     scale_factor = (a_max_block[:, None] * b_max_block[None, :]) / (127.0 * 127.0)
    
#     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    
#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#         a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
#         b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N), other=0.0)
        
#         accumulator += tl.dot(a, b)
        
#         a_ptrs += BLOCK_SIZE_K * stride_ak
#         b_ptrs += BLOCK_SIZE_K * stride_bk
    
#     # 应用正确的缩放因子
#     c = accumulator.to(tl.float32) * scale_factor
    
#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
#     mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#     tl.store(c_ptrs, c, mask=mask)

# def per_channel_scaled_int8_mm_triton(
#     A: torch.Tensor, B: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor
# ) -> torch.Tensor:
#     """
#     修复后的矩阵乘法函数
#     注意：现在a_scale和b_scale应该是原始的最大值，不是缩放因子
#     """
#     assert A.dtype == torch.int8, "A must be int8"
#     assert B.dtype == torch.int8, "B must be int8"
#     assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"
    
#     M, K = A.shape
#     K2, N = B.shape
#     assert K == K2, f"Matrix dimensions don't match: A[{M}, {K}] @ B[{K2}, {N}]"
    
#     # 修复：现在期望的是最大值，不是缩放因子
#     assert a_scale.shape == (M,), f"a_scale should have shape ({M},), got {a_scale.shape}"
#     assert b_scale.shape == (N,), f"b_scale should have shape ({N},), got {b_scale.shape}"
    
#     C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
#     a_scale = a_scale.to(A.device)
#     b_scale = b_scale.to(A.device)
    
#     grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
#     per_channel_scaled_int8_mm_kernel[grid](
#         A, B, C,
#         M, N, K,
#         A.stride(0), A.stride(1),
#         B.stride(0), B.stride(1), 
#         C.stride(0), C.stride(1),
#         a_scale, b_scale,
#     )
    
#     return C




# #################################################per_group_scaled_int8_mm_triton###########################################################
# @triton.jit
# def per_group_scaled_int8_mm_kernel(
#     # 输入张量指针
#     A_q_ptr,           # int8 [M, K]
#     B_q_ptr,           # int8 [K, N] 
#     a_scale_ptr,       # 缩放因子 [M, G] 
#     b_scale_ptr,       # 缩放因子 [G, N] 
#     output_ptr,        # float32 [M, N]
    
#     # 张量维度
#     M, N, K,
#     group_size: tl.constexpr,
    
#     # 块大小
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
    
#     # 分组相关
#     G: tl.constexpr,
    
#     # 内存布局
#     stride_am, stride_ak,
#     stride_bk, stride_bn,
#     stride_ag, stride_ag_inner,
#     stride_bg, stride_bn_inner,
#     stride_om, stride_on,
# ):
#     # 程序ID
#     pid = tl.program_id(0)
#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
#     pid_m = pid // num_pid_n
#     pid_n = pid % num_pid_n
    
#     # 计算输出块的偏移
#     offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
#     # 创建掩码防止越界访问
#     mask_m = offs_m < M
#     mask_n = offs_n < N
    
#     # 初始化累加器
#     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
#     # 循环处理所有分组
#     for g in range(G):
#         # 当前组的范围
#         k_start = g * group_size
#         k_end = min((g + 1) * group_size, K)
        
#         # 加载当前组的缩放因子
#         a_scale_g = tl.load(
#             a_scale_ptr + offs_m[:, None] * stride_ag + g * stride_ag_inner,
#             mask=mask_m[:, None],
#             other=1.0
#         )
        
#         b_scale_g = tl.load(
#             b_scale_ptr + g * stride_bg + offs_n[None, :] * stride_bn_inner,
#             mask=mask_n[None, :],
#             other=1.0
#         )
        
#         # 计算组合缩放因子
#         scale_factor = a_scale_g * b_scale_g
        
#         # 初始化当前组的累加器
#         group_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
        
#         # 在当前组内进行矩阵乘法
#         for k_block in range(0, group_size, BLOCK_SIZE_K):
#             k_offs = k_start + k_block + tl.arange(0, BLOCK_SIZE_K)
#             mask_k = k_offs < k_end
            
#             # 加载A的块 [BLOCK_SIZE_M, BLOCK_SIZE_K]
#             a_ptrs = (
#                 A_q_ptr + 
#                 offs_m[:, None] * stride_am + 
#                 k_offs[None, :] * stride_ak
#             )
#             a_block = tl.load(
#                 a_ptrs,
#                 mask=mask_m[:, None] & mask_k[None, :],
#                 other=0
#             )
            
#             # 加载B的块 [BLOCK_SIZE_K, BLOCK_SIZE_N]
#             b_ptrs = (
#                 B_q_ptr + 
#                 k_offs[:, None] * stride_bk + 
#                 offs_n[None, :] * stride_bn
#             )
#             b_block = tl.load(
#                 b_ptrs,
#                 mask=mask_k[:, None] & mask_n[None, :],
#                 other=0
#             )
            
#             # 累加矩阵乘法结果
#             group_accumulator += tl.dot(a_block, b_block)
        
#         # 应用缩放因子并累加到总结果
#         accumulator += group_accumulator.to(tl.float32) * scale_factor
    
#     # 将结果写回输出
#     out_ptrs = (
#         output_ptr + 
#         offs_m[:, None] * stride_om + 
#         offs_n[None, :] * stride_on
#     )
#     tl.store(
#         out_ptrs,
#         accumulator,
#         mask=mask_m[:, None] & mask_n[None, :]
#     )

# def per_group_scaled_int8_mm_triton(
#     A_q: torch.Tensor,
#     B_q: torch.Tensor,
#     a_scale: torch.Tensor,
#     b_scale: torch.Tensor,
#     group_size: int = 128
# ) -> torch.Tensor:
#     """
#     Triton实现的逐分组缩放int8矩阵乘法
    
#     Args:
#         A_q: int8量化矩阵 [M, K]
#         B_q: int8量化矩阵 [K, N] 
#         a_scale: A的缩放因子 [M, G]
#         b_scale: B的缩放因子 [G, N]
#         group_size: 分组大小
        
#     Returns:
#         output: float32结果矩阵 [M, N]
#     """
#     assert A_q.is_cuda and B_q.is_cuda
#     assert a_scale.is_cuda and b_scale.is_cuda
    
#     M, K = A_q.shape
#     _, N = B_q.shape
#     G = K // group_size
    
#     # 检查维度一致性
#     assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"
#     assert a_scale.shape == (M, G), f"a_scale shape mismatch: {a_scale.shape} vs expected {(M, G)}"
#     assert b_scale.shape == (G, N), f"b_scale shape mismatch: {b_scale.shape} vs expected {(G, N)}"
    
#     # 分配输出张量
#     output = torch.empty((M, N), dtype=torch.float32, device=A_q.device)
    
#     # 定义块大小
#     BLOCK_SIZE_M = 64
#     BLOCK_SIZE_N = 64
#     BLOCK_SIZE_K = 32
    
#     # 计算网格大小
#     grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
#     # 启动kernel
#     per_group_scaled_int8_mm_kernel[grid](
#         A_q, B_q, a_scale, b_scale, output,
#         M, N, K, group_size,
#         BLOCK_SIZE_M=BLOCK_SIZE_M,
#         BLOCK_SIZE_N=BLOCK_SIZE_N, 
#         BLOCK_SIZE_K=BLOCK_SIZE_K,
#         G=G,
#         # 内存步长
#         stride_am=A_q.stride(0), stride_ak=A_q.stride(1),
#         stride_bk=B_q.stride(0), stride_bn=B_q.stride(1),
#         stride_ag=a_scale.stride(0), stride_ag_inner=a_scale.stride(1),
#         stride_bg=b_scale.stride(0), stride_bn_inner=b_scale.stride(1),
#         stride_om=output.stride(0), stride_on=output.stride(1),
#     )
    
#     return output


#################################################per_group_scaled_int8_mm_triton###########################################################
@triton.jit
def per_group_scaled_int8_mm_kernel(
    # 输入张量指针
    A_q_ptr,           # int8 [M, K]
    B_q_ptr,           # int8 [K, N] 
    a_scale_ptr,       # 缩放因子 [M, G]
    b_scale_ptr,       # 缩放因子 [G, N]
    output_ptr,        # float32 [M, N]
    
    # 张量维度
    M, N, K,
    G: tl.constexpr,   # 分组数量
    group_size: tl.constexpr,  # 每组大小
    
    # 内存布局参数
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_ag, stride_ag_inner,
    stride_bg, stride_bn_inner,
    stride_om, stride_on,
    
    # 块大小配置 (带默认值)
    BLOCK_SIZE_M: tl.constexpr = 64,
    BLOCK_SIZE_N: tl.constexpr = 64,
    BLOCK_SIZE_K: tl.constexpr = 32,
):
    # 程序ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for g in range(G):
        k_start = g * group_size
        k_end = min((g + 1) * group_size, K)
        
        a_scale_g = tl.load(
            a_scale_ptr + offs_m[:, None] * stride_ag + g * stride_ag_inner,
            mask=mask_m[:, None],
            other=1.0
        )
        
        b_scale_g = tl.load(
            b_scale_ptr + g * stride_bg + offs_n[None, :] * stride_bn_inner,
            mask=mask_n[None, :],
            other=1.0
        )
        
        scale_factor = a_scale_g * b_scale_g
        
        group_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
        
        for k_block in range(0, group_size, BLOCK_SIZE_K):
            k_offs = k_start + k_block + tl.arange(0, BLOCK_SIZE_K)
            mask_k = k_offs < k_end
            
            a_ptrs = A_q_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
            a_block = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0)
            
            b_ptrs = B_q_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
            b_block = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0)
            
            group_accumulator += tl.dot(a_block, b_block)
        
        accumulator += group_accumulator.to(tl.float32) * scale_factor
    
    out_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, accumulator, mask=mask_m[:, None] & mask_n[None, :])

def per_group_scaled_int8_mm_triton(
    A_q: torch.Tensor,
    B_q: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    group_size: int = 128
) -> torch.Tensor:
    assert A_q.is_cuda and B_q.is_cuda
    assert a_scale.is_cuda and b_scale.is_cuda
    
    M, K = A_q.shape
    _, N = B_q.shape
    G = K // group_size
    
    assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"
    assert a_scale.shape == (M, G), f"a_scale shape mismatch: {a_scale.shape} vs expected {(M, G)}"
    assert b_scale.shape == (G, N), f"b_scale shape mismatch: {b_scale.shape} vs expected {(G, N)}"
    
    output = torch.empty((M, N), dtype=torch.float32, device=A_q.device)
    
    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)
    
    per_group_scaled_int8_mm_kernel[grid](
        A_q, B_q, a_scale, b_scale, output,
        M, N, K, G, group_size,
        stride_am=A_q.stride(0), stride_ak=A_q.stride(1),
        stride_bk=B_q.stride(0), stride_bn=B_q.stride(1),
        stride_ag=a_scale.stride(0), stride_ag_inner=a_scale.stride(1),
        stride_bg=b_scale.stride(0), stride_bn_inner=b_scale.stride(1),
        stride_om=output.stride(0), stride_on=output.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32
    )
    
    return output





# ##############################################per_rowgroup_block_scaled_int8_mm_triton###########################################################

# # 简化版本 - 不使用动态索引
# @triton.jit
# def per_rowgroup_block_scaled_int8_mm_kernel(
#     A_q_ptr, B_q_ptr, A_inv_scale_ptr, B_inv_scale_ptr, out_ptr,
#     M, N, K,
#     group_size, col_group_size,
#     num_k_groups, num_col_groups,
#     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
#     stride_am, stride_ak, stride_bk, stride_bn,
#     stride_scale_am, stride_scale_ag, stride_scale_bg, stride_scale_bc,
#     stride_out_m, stride_out_n,
# ):
#     pid_m = tl.program_id(0)
#     pid_n = tl.program_id(1)
#     pid_g = tl.program_id(2)  # K组索引
    
#     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
#     rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
#     # 当前K组的范围
#     k_start = pid_g * group_size
#     k_end = min((pid_g + 1) * group_size, K)
    
#     # 初始化当前K组的累加器
#     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
#     # 在当前K组内进行分块矩阵乘法
#     for k_block in range(0, group_size, BLOCK_K):
#         k_offs = k_start + k_block + tl.arange(0, BLOCK_K)
        
#         # 加载A和B的块
#         a_ptrs = A_q_ptr + rm[:, None] * stride_am + k_offs[None, :] * stride_ak
#         a_mask = (rm[:, None] < M) & (k_offs[None, :] < k_end)
#         a = tl.load(a_ptrs, mask=a_mask, other=0).to(tl.int8)
        
#         b_ptrs = B_q_ptr + k_offs[:, None] * stride_bk + rn[None, :] * stride_bn
#         b_mask = (k_offs[:, None] < k_end) & (rn[None, :] < N)
#         b = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.int8)
        
#         acc += tl.dot(a, b)
    
#     # 加载缩放因子
#     a_scale_ptrs = A_inv_scale_ptr + rm * stride_scale_am + pid_g * stride_scale_ag
#     a_scale_mask = rm < M
#     a_scale = tl.load(a_scale_ptrs, mask=a_scale_mask, other=1.0)
    
#     b_scale_ptrs = B_inv_scale_ptr + pid_g * stride_scale_bg + pid_n * stride_scale_bc
#     b_scale = tl.load(b_scale_ptrs)
    
#     # 应用缩放因子
#     scale = a_scale[:, None] * b_scale
#     scaled_acc = acc * scale
    
#     # 累加到输出
#     out_ptrs = out_ptr + rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
#     out_mask = (rm[:, None] < M) & (rn[None, :] < N)
#     tl.atomic_add(out_ptrs, scaled_acc, mask=out_mask)

# def per_rowgroup_block_scaled_int8_mm_triton(
#     A_q: torch.Tensor,        
#     B_q: torch.Tensor,        
#     A_inv_scale: torch.Tensor,
#     B_inv_scale: torch.Tensor,
#     row_group_size: int = 128,
#     col_group_size: int = 128,
#     group_size: int = 128,
#     BLOCK_M: int = 64,
#     BLOCK_N: int = 64,
#     BLOCK_K: int = 32
# ) -> torch.Tensor:
#     """
#     简化版本的Triton kernel实现
#     """
#     assert A_q.is_cuda and B_q.is_cuda
#     assert A_q.dtype == torch.int8 and B_q.dtype == torch.int8
    
#     M, K = A_q.shape
#     _, N = B_q.shape
    
#     # 验证维度
#     assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
#     assert N % col_group_size == 0, f"N={N} must be divisible by col_group_size={col_group_size}"
    
#     num_k_groups = K // group_size
#     num_col_groups = N // col_group_size
    
#     # 分配输出张量
#     out = torch.zeros((M, N), dtype=torch.float32, device=A_q.device)
    
#     # 计算网格大小 - 现在有3个维度
#     grid = (
#         triton.cdiv(M, BLOCK_M),
#         triton.cdiv(N, BLOCK_N),
#         num_k_groups
#     )
    
#     # 启动kernel
#     per_rowgroup_block_scaled_int8_mm_kernel[grid](
#         A_q, B_q, A_inv_scale, B_inv_scale, out,
#         M, N, K,
#         group_size, col_group_size,
#         num_k_groups, num_col_groups,
#         BLOCK_M, BLOCK_N, BLOCK_K,
#         A_q.stride(0), A_q.stride(1),
#         B_q.stride(0), B_q.stride(1), 
#         A_inv_scale.stride(0), A_inv_scale.stride(1),
#         B_inv_scale.stride(0), B_inv_scale.stride(1),
#         out.stride(0), out.stride(1),
#     )
    
#     return out


##############################################per_rowgroup_block_scaled_int8_mm_triton###########################################################

# @triton.jit
# def per_rowgroup_block_scaled_int8_mm_kernel(
#     A_q_ptr, B_q_ptr, A_scale_ptr, B_scale_ptr, out_ptr,  # 修复：参数名改为scale
#     M, N, K,
#     group_size, col_group_size,
#     num_k_groups, num_col_groups,
#     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
#     stride_am, stride_ak, stride_bk, stride_bn,
#     stride_scale_am, stride_scale_ag, stride_scale_bg, stride_scale_bc,
#     stride_out_m, stride_out_n,
# ):
#     pid_m = tl.program_id(0)
#     pid_n = tl.program_id(1)
#     pid_g = tl.program_id(2)
    
#     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
#     rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
#     k_start = pid_g * group_size
#     k_end = min((pid_g + 1) * group_size, K)
    
#     # 修复：使用float32累加器
#     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    
#     # 在当前K组内进行分块矩阵乘法
#     for k_block in range(0, group_size, BLOCK_K):
#         k_offs = k_start + k_block + tl.arange(0, BLOCK_K)
#         k_mask = k_offs < k_end
        
#         # 加载A的块
#         a_ptrs = A_q_ptr + rm[:, None] * stride_am + k_offs[None, :] * stride_ak
#         a_mask = (rm[:, None] < M) & (k_offs[None, :] < k_end)
#         a = tl.load(a_ptrs, mask=a_mask, other=0).to(tl.int8)
        
#         # 加载B的块  
#         b_ptrs = B_q_ptr + k_offs[:, None] * stride_bk + rn[None, :] * stride_bn
#         b_mask = (k_offs[:, None] < k_end) & (rn[None, :] < N)
#         b = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.int8)
        
#         # 矩阵乘法累加
#         acc += tl.dot(a, b)
    
#     # 修复：正确加载和应用缩放因子
#     # A的缩放因子是每行每组的 [M, num_k_groups]
#     a_scale_ptrs = A_scale_ptr + rm * stride_scale_am + pid_g * stride_scale_ag
#     a_scale_mask = rm < M
#     a_scale = tl.load(a_scale_ptrs, mask=a_scale_mask, other=1.0)
    
#     # B的缩放因子是每组每列的 [num_k_groups, num_col_groups]
#     b_scale_ptrs = B_scale_ptr + pid_g * stride_scale_bg + pid_n * stride_scale_bc
#     b_scale = tl.load(b_scale_ptrs)
    
#     # 修复：正确的反量化公式 - (int8_result * a_scale * b_scale) / (127.0 * 127.0)
#     # 因为量化时：q = round(x * 127.0 / max_value)
#     # 所以反量化时：x ≈ q * max_value / 127.0
#     scale_factor = (a_scale[:, None] * b_scale[None, :]) / (127.0 * 127.0)
#     result = acc.to(tl.float32) * scale_factor
    
#     # 写入结果
#     out_ptrs = out_ptr + rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
#     out_mask = (rm[:, None] < M) & (rn[None, :] < N)
#     tl.store(out_ptrs, result, mask=out_mask)


# def per_rowgroup_block_scaled_int8_mm_triton(
#     A_q: torch.Tensor,        
#     B_q: torch.Tensor,        
#     A_scale: torch.Tensor,
#     B_scale: torch.Tensor,
#     group_size: int = 128,
#     col_group_size: int = 128,
#     BLOCK_M: int = 64,
#     BLOCK_N: int = 64, 
#     BLOCK_K: int = 32
# ) -> torch.Tensor:
#     # 简化版本，固定BLOCK大小
#     M, K = A_q.shape
#     _, N = B_q.shape
    
#     num_k_groups = K // group_size
#     num_col_groups = N // col_group_size
    
#     out = torch.empty((M, N), dtype=torch.float32, device=A_q.device)
    
#     grid = (
#         triton.cdiv(M, BLOCK_M),
#         triton.cdiv(N, BLOCK_N), 
#         num_k_groups
#     )
    
#     per_rowgroup_block_scaled_int8_mm_kernel[grid](
#         A_q, B_q, A_scale, B_scale, out,
#         M, N, K,
#         group_size, col_group_size,
#         num_k_groups, num_col_groups,
#         BLOCK_M, BLOCK_N, BLOCK_K,
#         A_q.stride(0), A_q.stride(1),
#         B_q.stride(0), B_q.stride(1),
#         A_scale.stride(0), A_scale.stride(1),
#         B_scale.stride(0), B_scale.stride(1),
#         out.stride(0), out.stride(1),
#     )
    
#     return out



##############################################per_rowgroup_block_scaled_int8_mm_triton###########################################################



@triton.jit
def per_rowgroup_block_scaled_int8_mm_kernel(
    A_q_ptr, B_q_ptr, A_inv_scale_ptr, B_inv_scale_ptr, out_ptr,
    M, N, K,
    group_size, col_group_size,
    num_k_groups, num_col_groups,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_scale_am, stride_scale_ag, stride_scale_bg, stride_scale_bc,
    stride_out_m, stride_out_n,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算当前块的行/列索引
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 初始化浮点累加器（用于累加所有K组的结果）
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 遍历所有K组
    for pid_g in range(num_k_groups):
        k_start = pid_g * group_size
        k_end = tl.minimum((pid_g + 1) * group_size, K)
        
        # 初始化当前K组的整数累加器
        group_accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        
        # 在当前K组内分块计算
        for k_block in range(0, group_size, BLOCK_K):
            k_offs = k_start + k_block + tl.arange(0, BLOCK_K)
            k_mask = k_offs < k_end  # 确保不超出当前K组边界
            
            # 加载A块 [BLOCK_M, BLOCK_K]
            a_ptrs = A_q_ptr + rm[:, None] * stride_am + k_offs[None, :] * stride_ak
            a_mask = (rm[:, None] < M) & k_mask[None, :]
            a = tl.load(a_ptrs, mask=a_mask, other=0).to(tl.int8)
            
            # 加载B块 [BLOCK_K, BLOCK_N]
            b_ptrs = B_q_ptr + k_offs[:, None] * stride_bk + rn[None, :] * stride_bn
            b_mask = k_mask[:, None] & (rn[None, :] < N)
            b = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.int8)
            
            # 累加当前块结果
            group_accumulator += tl.dot(a, b)
        
        # 加载A的缩放因子 [BLOCK_M]
        a_scale_ptrs = A_inv_scale_ptr + rm * stride_scale_am + pid_g * stride_scale_ag
        a_scale_mask = rm < M
        a_scale = tl.load(a_scale_ptrs, mask=a_scale_mask, other=1.0)
        
        # 为每列独立计算列组索引
        col_group_idx = rn // col_group_size
        col_group_idx = tl.minimum(col_group_idx, num_col_groups - 1)
        
        # 加载B的缩放因子 [BLOCK_N] (每个列组一个值)
        b_scale_ptrs = B_inv_scale_ptr + pid_g * stride_scale_bg + col_group_idx * stride_scale_bc
        b_scale_mask = rn < N
        b_scale = tl.load(b_scale_ptrs, mask=b_scale_mask, other=1.0)
        
        # 计算缩放因子 [BLOCK_M, BLOCK_N]
        scale_factor = a_scale[:, None] * b_scale[None, :]
        
        # 将整数结果转换为浮点并应用缩放
        group_result = group_accumulator.to(tl.float32) * scale_factor
        
        # 累加到总结果
        accumulator += group_result
    
    # 写入最终结果
    out_ptrs = out_ptr + rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
    mask_out = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=mask_out)

def per_rowgroup_block_scaled_int8_mm_triton(
    A_q: torch.Tensor,
    B_q: torch.Tensor,
    A_inv_scale: torch.Tensor,
    B_inv_scale: torch.Tensor,
    group_size: int = 128,
    col_group_size: int = 128,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32
) -> torch.Tensor:
    M, K = A_q.shape
    _, N = B_q.shape
    
    # 验证输入形状
    assert K % group_size == 0, f"K dimension {K} must be divisible by group_size {group_size}"
    assert N % col_group_size == 0, f"N dimension {N} must be divisible by col_group_size {col_group_size}"
    
    num_k_groups = K // group_size
    num_col_groups = N // col_group_size
    
    # 验证缩放张量形状
    assert A_inv_scale.shape == (M, num_k_groups), \
        f"A_inv_scale shape mismatch: {A_inv_scale.shape} vs expected {(M, num_k_groups)}"
    assert B_inv_scale.shape == (num_k_groups, num_col_groups), \
        f"B_inv_scale shape mismatch: {B_inv_scale.shape} vs expected {(num_k_groups, num_col_groups)}"
    
    # 创建输出张量
    out = torch.empty((M, N), dtype=torch.float32, device=A_q.device)
    
    # 配置grid (移除K组维度)
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )
    
    # 启动内核
    per_rowgroup_block_scaled_int8_mm_kernel[grid](
        A_q, B_q, A_inv_scale, B_inv_scale, out,
        M, N, K,
        group_size, col_group_size,
        num_k_groups, num_col_groups,
        BLOCK_M, BLOCK_N, BLOCK_K,
        A_q.stride(0), A_q.stride(1),
        B_q.stride(0), B_q.stride(1),
        A_inv_scale.stride(0), A_inv_scale.stride(1),
        B_inv_scale.stride(0), B_inv_scale.stride(1),
        out.stride(0), out.stride(1),
    )
    
    return out





##################################################per_block_scaled_int8_mm_triton##########################################################
@triton.jit
def per_block_scaled_int8_mm_kernel(
    # 输入张量指针
    A_q_ptr,           # int8 [M, K]
    B_q_ptr,           # int8 [K, N] 
    a_scale_ptr,       # 缩放因子 [M, G] 
    b_scale_ptr,       # 缩放因子 [G, N] 
    output_ptr,        # float32 [M, N]
    
    # 张量维度
    M, N, K,
    group_size: tl.constexpr,
    
    # 块大小
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    
    # 分组相关
    G: tl.constexpr,
    
    # 内存布局
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_ag, stride_ag_inner,
    stride_bg, stride_bn_inner,
    stride_om, stride_on,
):
    # 程序ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # 计算输出块的偏移
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # 创建掩码防止越界访问
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 循环处理所有分组
    for g in range(G):
        # 当前组的范围
        k_start = g * group_size
        k_end = min((g + 1) * group_size, K)
        
        # 加载当前组的缩放因子
        a_scale_g = tl.load(
            a_scale_ptr + offs_m[:, None] * stride_ag + g * stride_ag_inner,
            mask=mask_m[:, None],
            other=1.0
        )
        
        b_scale_g = tl.load(
            b_scale_ptr + g * stride_bg + offs_n[None, :] * stride_bn_inner,
            mask=mask_n[None, :],
            other=1.0
        )
        
        # 计算组合缩放因子
        scale_factor = a_scale_g * b_scale_g
        
        # 初始化当前组的累加器
        group_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
        
        # 在当前组内进行矩阵乘法
        for k_block in range(0, group_size, BLOCK_SIZE_K):
            k_offs = k_start + k_block + tl.arange(0, BLOCK_SIZE_K)
            mask_k = k_offs < k_end
            
            # 加载A的块 [BLOCK_SIZE_M, BLOCK_SIZE_K]
            a_ptrs = (
                A_q_ptr + 
                offs_m[:, None] * stride_am + 
                k_offs[None, :] * stride_ak
            )
            a_block = tl.load(
                a_ptrs,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0
            )
            
            # 加载B的块 [BLOCK_SIZE_K, BLOCK_SIZE_N]
            b_ptrs = (
                B_q_ptr + 
                k_offs[:, None] * stride_bk + 
                offs_n[None, :] * stride_bn
            )
            b_block = tl.load(
                b_ptrs,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0
            )
            
            # 累加矩阵乘法结果
            group_accumulator += tl.dot(a_block, b_block)
        
        # 应用缩放因子并累加到总结果
        accumulator += group_accumulator.to(tl.float32) * scale_factor
    
    # 将结果写回输出
    out_ptrs = (
        output_ptr + 
        offs_m[:, None] * stride_om + 
        offs_n[None, :] * stride_on
    )
    tl.store(
        out_ptrs,
        accumulator,
        mask=mask_m[:, None] & mask_n[None, :]
    )

def per_block_scaled_int8_mm_triton(
    A_q: torch.Tensor,
    B_q: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    group_size: int = 128
) -> torch.Tensor:
    """
    简化版本的逐块缩放int8矩阵乘法
    
    Args:
        A_q: int8量化矩阵 [M, K]
        B_q: int8量化矩阵 [K, N] 
        a_scale: A的缩放因子 [M, G]
        b_scale: B的缩放因子 [G, N]
        group_size: 分组大小
        
    Returns:
        output: float32结果矩阵 [M, N]
    """
    assert A_q.is_cuda and B_q.is_cuda
    assert a_scale.is_cuda and b_scale.is_cuda
    
    M, K = A_q.shape
    _, N = B_q.shape
    G = K // group_size
    
    # 检查维度一致性
    assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"
    assert a_scale.shape == (M, G), f"a_scale shape mismatch: {a_scale.shape} vs expected {(M, G)}"
    assert b_scale.shape == (G, N), f"b_scale shape mismatch: {b_scale.shape} vs expected {(G, N)}"
    
    # 分配输出张量
    output = torch.empty((M, N), dtype=torch.float32, device=A_q.device)
    
    # 定义块大小
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # 计算网格大小
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    # 启动kernel
    per_block_scaled_int8_mm_kernel[grid](
        A_q, B_q, a_scale, b_scale, output,
        M, N, K, group_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N, 
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        G=G,
        # 内存步长
        stride_am=A_q.stride(0), stride_ak=A_q.stride(1),
        stride_bk=B_q.stride(0), stride_bn=B_q.stride(1),
        stride_ag=a_scale.stride(0), stride_ag_inner=a_scale.stride(1),
        stride_bg=b_scale.stride(0), stride_bn_inner=b_scale.stride(1),
        stride_om=output.stride(0), stride_on=output.stride(1),
    )
    
    return output


##################################################per_token_tensor_scaled_int8_mm_triton##########################################################

@triton.jit
def per_token_tensor_scaled_int8_mm_kernel(
    # 输入张量指针
    A_q_ptr,           # int8 [M, K] - 行量化
    B_q_ptr,           # int8 [K, N] - 张量量化
    a_scale_ptr,       # 缩放因子 [M] - 每行的缩放因子
    b_scale_ptr,       # 缩放因子 [1] - 整个张量的缩放因子
    output_ptr,        # float32 [M, N]
    
    # 张量维度
    M, N, K,
    
    # 块大小
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    
    # 内存布局
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_as,  # a_scale的步长
    stride_bs,  # b_scale的步长
    stride_om, stride_on,
):
    # 程序ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # 计算输出块的偏移
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # 创建掩码防止越界访问
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # 加载当前块的缩放因子
    # a_scale是每行的缩放因子 [M]
    a_scale_block = tl.load(
        a_scale_ptr + offs_m * stride_as,
        mask=mask_m,
        other=1.0
    )
    
    # b_scale是整个张量的缩放因子 [1]
    b_scale_val = tl.load(b_scale_ptr)
    
    # 计算组合缩放因子 [BLOCK_SIZE_M, 1]
    scale_factor = a_scale_block[:, None] * b_scale_val
    
    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 循环处理K维度
    for k in range(0, K, BLOCK_SIZE_K):
        k_offs = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offs < K
        
        # 加载A的块 [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_ptrs = (
            A_q_ptr + 
            offs_m[:, None] * stride_am + 
            k_offs[None, :] * stride_ak
        )
        a_block = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0
        )
        
        # 加载B的块 [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b_ptrs = (
            B_q_ptr + 
            k_offs[:, None] * stride_bk + 
            offs_n[None, :] * stride_bn
        )
        b_block = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0
        )
        
        # 累加矩阵乘法结果
        accumulator += tl.dot(a_block, b_block)
    
    # 应用缩放因子
    accumulator = accumulator * scale_factor
    
    # 将结果写回输出
    out_ptrs = (
        output_ptr + 
        offs_m[:, None] * stride_om + 
        offs_n[None, :] * stride_on
    )
    tl.store(
        out_ptrs,
        accumulator,
        mask=mask_m[:, None] & mask_n[None, :]
    )

def per_token_tensor_scaled_int8_mm_triton(
    A_q: torch.Tensor,
    B_q: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor
) -> torch.Tensor:
    """
    Triton实现的token-wise和tensor-wise混合量化矩阵乘法
    
    Args:
        A_q: int8量化矩阵 [M, K] - 行量化
        B_q: int8量化矩阵 [K, N] - 张量量化
        a_scale: A的缩放因子 [M] - 每行的缩放因子
        b_scale: B的缩放因子 [1] - 整个张量的缩放因子
        
    Returns:
        output: float32结果矩阵 [M, N]
    """
    assert A_q.is_cuda and B_q.is_cuda
    assert a_scale.is_cuda and b_scale.is_cuda
    
    M, K = A_q.shape
    _, N = B_q.shape
    
    # 检查维度一致性
    assert a_scale.shape == (M,), f"a_scale shape mismatch: {a_scale.shape} vs expected {(M,)}"
    assert b_scale.numel() == 1, f"b_scale should be a scalar, got shape {b_scale.shape}"
    
    # 分配输出张量
    output = torch.empty((M, N), dtype=torch.float32, device=A_q.device)
    
    # 定义块大小
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # 计算网格大小
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    # 启动kernel
    per_token_tensor_scaled_int8_mm_kernel[grid](
        A_q, B_q, a_scale, b_scale, output,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N, 
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        # 内存步长
        stride_am=A_q.stride(0), stride_ak=A_q.stride(1),
        stride_bk=B_q.stride(0), stride_bn=B_q.stride(1),
        stride_as=a_scale.stride(0),
        stride_bs=0,  # b_scale是标量，步长为0
        stride_om=output.stride(0), stride_on=output.stride(1),
    )
    
    return output









##################################################per_tensor_token_scaled_int8_mm_triton##########################################################


@triton.jit
def per_tensor_token_scaled_int8_mm_kernel(
    # 输入张量指针
    A_q_ptr,           # int8 [M, K] - 张量量化
    B_q_ptr,           # int8 [K, N] - 列量化
    a_scale_ptr,       # 缩放因子 [1] - 整个张量的缩放因子
    b_scale_ptr,       # 缩放因子 [N] - 每列的缩放因子
    output_ptr,        # float32 [M, N]
    
    # 张量维度
    M, N, K,
    
    # 块大小
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    
    # 内存布局
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_as,  # a_scale的步长
    stride_bs,  # b_scale的步长
    stride_om, stride_on,
):
    # 程序ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # 计算输出块的偏移
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # 创建掩码防止越界访问
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # 加载当前块的缩放因子
    # a_scale是整个张量的缩放因子 [1]
    a_scale_val = tl.load(a_scale_ptr)
    
    # b_scale是每列的缩放因子 [N]
    b_scale_block = tl.load(
        b_scale_ptr + offs_n * stride_bs,
        mask=mask_n,
        other=1.0
    )
    
    # 计算组合缩放因子 [1, BLOCK_SIZE_N]
    scale_factor = a_scale_val * b_scale_block[None, :]
    
    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 循环处理K维度
    for k in range(0, K, BLOCK_SIZE_K):
        k_offs = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offs < K
        
        # 加载A的块 [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_ptrs = (
            A_q_ptr + 
            offs_m[:, None] * stride_am + 
            k_offs[None, :] * stride_ak
        )
        a_block = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0
        )
        
        # 加载B的块 [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b_ptrs = (
            B_q_ptr + 
            k_offs[:, None] * stride_bk + 
            offs_n[None, :] * stride_bn
        )
        b_block = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0
        )
        
        # 累加矩阵乘法结果
        accumulator += tl.dot(a_block, b_block)
    
    # 应用缩放因子
    accumulator = accumulator * scale_factor
    
    # 将结果写回输出
    out_ptrs = (
        output_ptr + 
        offs_m[:, None] * stride_om + 
        offs_n[None, :] * stride_on
    )
    tl.store(
        out_ptrs,
        accumulator,
        mask=mask_m[:, None] & mask_n[None, :]
    )

def per_tensor_token_scaled_int8_mm_triton(
    A_q: torch.Tensor,
    B_q: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor
) -> torch.Tensor:
    """
    Triton实现的tensor-wise和token-wise混合量化矩阵乘法
    
    Args:
        A_q: int8量化矩阵 [M, K] - 张量量化
        B_q: int8量化矩阵 [K, N] - 列量化
        a_scale: A的缩放因子 [1] - 整个张量的缩放因子
        b_scale: B的缩放因子 [N] - 每列的缩放因子
        
    Returns:
        output: float32结果矩阵 [M, N]
    """
    assert A_q.is_cuda and B_q.is_cuda
    assert a_scale.is_cuda and b_scale.is_cuda
    
    M, K = A_q.shape
    _, N = B_q.shape
    
    # 检查维度一致性
    assert a_scale.numel() == 1, f"a_scale should be a scalar, got shape {a_scale.shape}"
    assert b_scale.shape == (N,), f"b_scale shape mismatch: {b_scale.shape} vs expected {(N,)}"
    
    # 分配输出张量
    output = torch.empty((M, N), dtype=torch.float32, device=A_q.device)
    
    # 定义块大小
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # 计算网格大小
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    # 启动kernel
    per_tensor_token_scaled_int8_mm_kernel[grid](
        A_q, B_q, a_scale, b_scale, output,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N, 
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        # 内存步长
        stride_am=A_q.stride(0), stride_ak=A_q.stride(1),
        stride_bk=B_q.stride(0), stride_bn=B_q.stride(1),
        stride_as=0,  # a_scale是标量，步长为0
        stride_bs=b_scale.stride(0),
        stride_om=output.stride(0), stride_on=output.stride(1),
    )
    
    return output

# ==================== 矩阵乘法实现 ====================

############################################# per_tensor矩阵乘法 ###############################################



def _select_int8_ops_triton(mode):
    if mode == "tensor":  # tensor * tensor
        return per_tensor_quantize_int8_triton, per_tensor_quantize_int8_triton, per_tensor_scaled_int8_mm_triton
    elif mode == "channel":  # token * token
        return per_row_quantize_int8_triton, per_col_quantize_int8_triton, per_channel_scaled_int8_mm_triton
    elif mode == "tile":  # tile * tile
        return per_group_row_quantize_int8_triton, per_group_col_quantize_int8_triton, per_group_scaled_int8_mm_triton
    elif mode == "tile_block":  # tile * block
        return per_group_row_quantize_int8_triton, per_block_quantize_int8_triton, per_rowgroup_block_scaled_int8_mm
    elif mode == "block":  # block * block
        return per_block_quantize_int8_triton, per_block_quantize_int8_triton, per_block_scaled_int8_mm_triton
    elif mode == "channel_tensor":  # token * tensor
        return per_row_quantize_int8_triton, per_tensor_quantize_int8_triton, per_token_tensor_scaled_int8_mm_triton
    elif mode == "tensor_channel":  # tensor * token
        return per_tensor_quantize_int8_triton, per_col_quantize_int8_triton, per_tensor_token_scaled_int8_mm_triton
    else:
        raise ValueError(f"Unsupported mode: {mode}")




from torchao.prototype.quantized_training.int8_mm import scaled_int8_mm, scaled_int8_mm_cuda
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

    return x_scaled, x_amax #/ 127


def per_group_row_quantize_int8(x: torch.Tensor, sr=True, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    M, K = x.shape
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

    return x_scaled, x_amax #/ 127


def per_block_quantize_int8(x: torch.Tensor, sr: bool = False, row_group_size: int = 128, col_group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    M, K = x.shape
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

    # inv_scale = block_amax / 127.0
    return x_q, block_amax#inv_scale

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

def quantized_matmul(a: torch.Tensor, b: torch.Tensor, mode: str = "tensor", group_size: int = 128):
    """量化矩阵乘法主函数"""
    if mode == "tensor":
        # 原有的张量量化逻辑
        a_int8, a_max = per_tensor_quantize_int8_triton(a,False)
        b_int8, b_max = per_tensor_quantize_int8_triton(b,False)

        a_scale = a_max / 127.0
        b_scale = b_max / 127.0

        mm_fn = per_tensor_scaled_int8_mm_triton
        result = mm_fn(a_int8, b_int8, a_scale, b_scale)
        
    elif mode == "channel":
        # 原有的通道量化逻辑
        a_int8, a_max = per_row_quantize_int8_triton(a,False)
        b_int8, b_max = per_col_quantize_int8_triton(b,False)

        a_scale = a_max / 127.0
        b_scale = b_max / 127.0

        mm_fn = per_channel_scaled_int8_mm_triton
        result = mm_fn(a_int8, b_int8, a_scale, b_scale)
        
    elif mode == "tile":
        # 修正的分组量化逻辑
        M, K = a.shape
        _, N = b.shape
        G = K // group_size
        
        # 量化输入矩阵
        a_int8, a_max = per_group_row_quantize_int8_triton(a, group_size=group_size,sr=False)
        b_int8, b_max = per_group_col_quantize_int8_triton(b, group_size=group_size,sr=False)
        
        a_scale = a_max / 127.0
        b_scale = b_max / 127.0

        # 执行量化矩阵乘法
        result = per_group_scaled_int8_mm_triton(a_int8, b_int8, a_scale, b_scale, group_size)
        
    elif mode == "tile_block":
        # 修正的分组量化逻辑
        M, K = a.shape
        _, N = b.shape
        G = K // group_size
        
        # 量化输入矩阵
        a_int8, a_max = per_group_row_quantize_int8_triton(a, group_size=group_size,sr=False)
        b_int8, b_max = per_block_quantize_int8_triton(b, (group_size,group_size),sr=False)
        
        a_scale = a_max / 127.0
        b_scale = b_max / 127.0

        # 执行量化矩阵乘法
        result = per_rowgroup_block_scaled_int8_mm_triton(a_int8, b_int8, a_scale, b_scale, group_size)

    elif mode == "block":
        # 块量化逻辑 - 使用相同的分组大小
        M, K = a.shape
        _, N = b.shape
        G = K // group_size
        
        # 量化输入矩阵
        a_int8, a_max = per_block_quantize_int8_triton(a, (group_size,group_size),sr=False)
        b_int8, b_max = per_block_quantize_int8_triton(b, (group_size,group_size),sr=False)
            
        # 应用缩放因子
        a_scale = a_max / 127.0
        b_scale = b_max / 127.0
        
        # 修正：只传递一个group_size参数
        result = per_block_scaled_int8_mm_triton(a_int8, b_int8, a_scale, b_scale, group_size)
    elif mode == "channel_tensor":
        # 块量化逻辑 - 使用相同的分组大小
        M, K = a.shape
        _, N = b.shape
        # G = K // group_size
        
        # 量化输入矩阵
        a_int8, a_max = per_row_quantize_int8_triton(a,False)
        b_int8, b_max = per_tensor_quantize_int8_triton(b,False)
            
        # 应用缩放因子
        a_scale = a_max / 127.0
        b_scale = b_max / 127.0
        
        # 修正：只传递一个group_size参数
        result = per_token_tensor_scaled_int8_mm_triton(a_int8, b_int8, a_scale, b_scale)
    elif mode == "tensor_channel":
        # 块量化逻辑 - 使用相同的分组大小
        M, K = a.shape
        _, N = b.shape
        # G = K // group_size
        
        # 量化输入矩阵
        a_int8, a_max = per_tensor_quantize_int8_triton(a,False)
        b_int8, b_max = per_col_quantize_int8_triton(b,False)
            
        # 应用缩放因子
        a_scale = a_max / 127.0
        b_scale = b_max / 127.0
        
        # 修正：只传递一个group_size参数
        result = per_tensor_token_scaled_int8_mm_triton(a_int8, b_int8, a_scale, b_scale)

    else:
        raise ValueError(f"不支持的量化模式: {mode}")
    
    return result


