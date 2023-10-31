/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "common.h"
#include "definitions.h"
#include "handle.h"
#include "rocsparse.h"
#include "rocsparse_reduce.hpp"
#include "rocsparse_sddmm.hpp"
#include "utility.h"

template <rocsparse_int       BLOCKSIZE,
          rocsparse_int       NTHREADS_PER_DOTPRODUCT,
          rocsparse_direction DIRECTION,
          typename I,
          typename J,
          typename T,
          typename U>
// ROCSPARSE_KERNEL是宏定义，通常用于标记GPU核函数（kernel）
// __launch_bounds__(BLOCKSIZE, 1)指示 GPU 核心函数的启动参数，即指定BLOCKSIZE为1（每个GPU块（block）中的线程数目）
ROCSPARSE_KERNEL __launch_bounds__(BLOCKSIZE,
                                   1) void sddmm_csx_kernel(rocsparse_operation transA,  // 操作的方向 transA、transB
                                                            rocsparse_operation transB,
                                                            rocsparse_order     orderA,
                                                            rocsparse_order     orderB,
                                                            J                   M,  // 矩阵大小 M N K
                                                            J                   N,
                                                            J                   K,
                                                            I                   nnz,  // 非零元数量nnz
                                                            U                   alpha_device_host,  // alpha 参数
                                                            const T* __restrict__ A,  // 输入矩阵 A、B
                                                            J lda,
                                                            const T* __restrict__ B,
                                                            J ldb,
                                                            U beta_device_host,  // beta 参数
                                                            T* __restrict__ csx_val,  // 输出矩阵 csx_val
                                                            const I* __restrict__ csx_ptr,
                                                            const J* __restrict__ csx_ind,
                                                            rocsparse_index_base csx_base,
                                                            T* __restrict__ workspace)
                                                            // 如果 alpha 等于0且 beta 等于1，函数会提前退出，因为在这种情况下不需要执行SDDMM操作。
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);
    if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
    {
        return;
    }

    //
    // Each group treats one row/column.
    //
    static constexpr rocsparse_int NUM_SEQS           = (BLOCKSIZE / NTHREADS_PER_DOTPRODUCT);
    const I                        local_seq_index    = hipThreadIdx_x / NTHREADS_PER_DOTPRODUCT;
    const I                        local_thread_index = hipThreadIdx_x % NTHREADS_PER_DOTPRODUCT;
    const I                        tid                = hipBlockIdx_x * NUM_SEQS + local_seq_index;
    static constexpr bool          row_oriented       = (DIRECTION == rocsparse_direction_row);
#define BOUND ((row_oriented) ? M : N)
    if(tid >= BOUND)
    {
        return;
    }

    const J incx = (orderA == rocsparse_order_column)
                       ? ((transA == rocsparse_operation_none) ? lda : 1)
                       : ((transA == rocsparse_operation_none) ? 1 : lda);

    const J incy = (orderB == rocsparse_order_column)
                       ? ((transB == rocsparse_operation_none) ? 1 : ldb)
                       : ((transB == rocsparse_operation_none) ? ldb : 1);

    const J xinc = (row_oriented) ? incx : incy; // 如果row_oriented为真，则表示按行存储
    const J yinc = (row_oriented) ? incy : incx;

    __shared__ T s[NUM_SEQS][NTHREADS_PER_DOTPRODUCT]; // 二维数组，NUM_SEQS行，NTHREADS_PER_DOTPRODUCT列，用于存储计算的中间结果

    // 定义了一个指针x，用于指向输入向量中的元素
    const T* x = (row_oriented) // row_oriented：矩阵乘法是否按行进行
                     ? ((orderA == rocsparse_order_column) // rocsparse_order_column：A是否按行存储
                            ? ((transA == rocsparse_operation_none) ? (A + tid) : (A + lda * tid)) // rocsparse_operation_none：矩阵A是否进行了转置
                            : ((transA == rocsparse_operation_none) ? (A + lda * tid) : (A + tid))) // tid：线程在线程块内的编号
                     : ((orderB == rocsparse_order_column)
                            ? ((transB == rocsparse_operation_none) ? (B + ldb * tid) : (B + tid))
                            : ((transB == rocsparse_operation_none) ? (B + tid) : (B + ldb * tid)));

    /* 核心代码：进行SDDMM的计算（Sparse-Dense-Dense Matrix Multiplication）*/
    // csx_ptr：行偏移数组。csx_ind：列索引。csx_base：枚举值，rocsparse_index_base_zero表示索引从0开始，rocsparse_index_base_one表示索引从1开始
    for(I at = csx_ptr[tid] - csx_base; at < csx_ptr[tid + 1] - csx_base; ++at) // 遍历稀疏矩阵的非零元素
    {
        I        ind = csx_ind[at] - csx_base; // 表示非零元在列号csx_ind中的索引位置。这个索引用于访问稠密矩阵 B 或 A 中的相应元素。
        const T* y //  用于指向稠密矩阵中的一列（列号？）
            = (row_oriented)
                  ? ((orderB == rocsparse_order_column)
                         ? ((transB == rocsparse_operation_none) ? (B + ldb * ind) : (B + ind))
                         : ((transB == rocsparse_operation_none) ? (B + ind) : (B + ldb * ind)))
                  : ((orderA == rocsparse_order_column)
                         ? ((transA == rocsparse_operation_none) ? (A + ind) : (A + lda * ind))
                         : ((transA == rocsparse_operation_none) ? (A + lda * ind) : (A + ind)));

        /* 核心代码：计算乘法操作*/
        T sum = static_cast<T>(0); // 初始化了一个变量 sum ，用于累积乘法的结果。
        for(J k = local_thread_index; k < K; k += NTHREADS_PER_DOTPRODUCT) 
        {
            sum += x[k * xinc] * y[k * yinc]; // 每一步的计算
        }
        s[local_seq_index][local_thread_index] = sum; // 将中间结果存在共享内存s中
        __syncthreads(); // 确保所有线程在继续执行前等待，从而保证共享内存中的所有数据都已更新。

// 一个编译器指令，用于提示编译器在循环内部进行循环展开（loop unrolling）：通过将循环体中的多个迭代合并成单个迭代来提高代码的执行效率。
#pragma unroll
    // 并行归约（parallel reduction）操作：将每个线程组内的中间结果相加并存储在 s 数组中。这个归约操作通常用于提高计算效率。
        for(int ipow2_ = 2; ipow2_ <= NTHREADS_PER_DOTPRODUCT; ipow2_ *= 2) // ipow2_从2开始，每次循环翻倍
        {
            if(local_thread_index < NTHREADS_PER_DOTPRODUCT / ipow2_) // 确保只有一半的线程参与到累加中，而另一半线程等待。
            {
                s[local_seq_index][local_thread_index]
                    += s[local_seq_index][local_thread_index + NTHREADS_PER_DOTPRODUCT / ipow2_]; // 每个线程会将共享内存数组 s 中的值与相邻线程的值相加。
            }
            __syncthreads(); // 同步点
        }

        // 最后的条件语句 if(local_thread_index == 0) 用于确保只有每个线程组内的一个线程（通常是第一个线程，即 local_thread_index == 0）执行以下操作：
        if(local_thread_index == 0)
        {
            csx_val[at] = csx_val[at] * beta + alpha * s[local_seq_index][0]; // 乘以 beta 并加上 alpha 与中间结果的乘积，并将其存储在稀疏矩阵 csx_val 中。
        }
    }
}
