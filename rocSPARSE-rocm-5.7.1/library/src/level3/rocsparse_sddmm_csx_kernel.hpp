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
// alpha通常是浮点数，在 SDDMM 操作中，每个非零元素的乘积都会与 alpha 相乘，以便调整结果的大小或权重。如果 alpha 为零，那么结果中的所有元素都将为零。
// 在 SDDMM 操作中，结果将与 beta 相乘，并且与原始结果相加。如果 beta 为零，那么结果将完全由 alpha 和 SDDMM 计算的新值构成。如果 beta 为1，那么新值将完全替代原始值。
ROCSPARSE_KERNEL __launch_bounds__(BLOCKSIZE,
                                   1) void sddmm_csx_kernel(rocsparse_operation transA,  // 枚举类型，rocsparse_operation_none表示不转置，rocsparse_operation_transpose表示转置
                                                            rocsparse_operation transB,
                                                            rocsparse_order     orderA, // 值为rocsparse_order_column表示列主序，rocsparse_order_row表示行主序
                                                            rocsparse_order     orderB,
                                                            J                   M,  // 矩阵大小 M N K
                                                            J                   N,
                                                            J                   K,
                                                            I                   nnz,  // 非零元数量nnz
                                                            U                   alpha_device_host,  // alpha 参数
                                                            const T* __restrict__ A,  // 输入矩阵 A、B
                                                            J lda, // 矩阵 A 的列步幅（leading dimension）：表示存储矩阵 A 中相邻列之间在内存中的间隔（偏移）。
                                                            const T* __restrict__ B, // __restrict__表示这个指针所指向的内存区域在其生命周期内不会被其他指针访问。
                                                            J ldb,
                                                            U beta_device_host,  // beta 参数
                                                            T* __restrict__ csx_val,  // 存储输出矩阵的非零元的值
                                                            const I* __restrict__ csx_ptr, // 指针（CSR里就是行偏移）
                                                            const J* __restrict__ csx_ind, // 索引（CSR里就是列号）
                                                            rocsparse_index_base csx_base, // 索引基准。若值为rocsparse_index_base_zero，则行列索引都从0开始，若为rocsparse_index_base_one则都从1开始
                                                            T* __restrict__ workspace) // 表示一块内存区域，用于暂时存储中间结果或临时数据。
{
    auto alpha = load_scalar_device_host(alpha_device_host); // 从主机内存中加载 alpha_device_host 和 beta_device_host 的值，并将它们分别存储到 alpha 和 beta 变量中。
    auto beta  = load_scalar_device_host(beta_device_host); 
    if(alpha == static_cast<T>(0) && beta == static_cast<T>(1)) // 如果 alpha 等于0且 beta 等于1，函数会提前退出，因为在这种情况下不需要执行SDDMM操作。
    {
        return;
    }

    // 
    // Each group treats one row/column.(每组处理一行或者一列)
    //为每个线程组（group）分配稀疏矩阵乘法（SDDMM）的计算任务
    static constexpr rocsparse_int NUM_SEQS           = (BLOCKSIZE / NTHREADS_PER_DOTPRODUCT); // 常量，表示每个线程组中的线程数。 BLOCKSIZE（线程块的大小）除以 NTHREADS_PER_DOTPRODUCT（每个点积计算中的线程数）。
    const I                        local_seq_index    = hipThreadIdx_x / NTHREADS_PER_DOTPRODUCT; //local_seq_index是每个线程在其所在线程组（group）中的序号，每个线程组要处理稀疏矩阵的一行或一列。 // hipThreadIdx_x是线程在线程块（block）内的索引，通常从0开始
    const I                        local_thread_index = hipThreadIdx_x % NTHREADS_PER_DOTPRODUCT; // local_thread_index 表示每个线程在其所在线程组内部的相对序号。
    const I                        tid                = hipBlockIdx_x * NUM_SEQS + local_seq_index; // 当前线程在整个网格（grid）中的全局唯一标识符。它是通过将 hipBlockIdx_x（线程块在网格中的序号）与 local_seq_index 相乘加上 local_thread_index 得到的。
    static constexpr bool          row_oriented       = (DIRECTION == rocsparse_direction_row); // 指示稀疏矩阵操作是按行还是按列进行的。这决定了在后续计算中如何选择增量值 incx 和 incy。
#define BOUND ((row_oriented) ? M : N) // 如果矩阵是按行计算，则BOUND为M（行数）；若按列计算，则BOUDN等于列数N
    if(tid >= BOUND)
    {
        return;
    }

    const J incx = (orderA == rocsparse_order_column) 
                       ? ((transA == rocsparse_operation_none) ? lda : 1) // 如果是列主序并进行了转置，则
                       : ((transA == rocsparse_operation_none) ? 1 : lda); // 如果是行主序：

    const J incy = (orderB == rocsparse_order_column)
                       ? ((transB == rocsparse_operation_none) ? 1 : ldb)
                       : ((transB == rocsparse_operation_none) ? ldb : 1);

    const J xinc = (row_oriented) ? incx : incy;
    const J yinc = (row_oriented) ? incy : incx;

    __shared__ T s[NUM_SEQS][NTHREADS_PER_DOTPRODUCT]; // 声明了一个共享内存数组 s，用于存储线程块内的局部中间结果。

    const T* x = (row_oriented)
                     ? ((orderA == rocsparse_order_column)
                            ? ((transA == rocsparse_operation_none) ? (A + tid) : (A + lda * tid))
                            : ((transA == rocsparse_operation_none) ? (A + lda * tid) : (A + tid)))
                     : ((orderB == rocsparse_order_column)
                            ? ((transB == rocsparse_operation_none) ? (B + ldb * tid) : (B + tid))
                            : ((transB == rocsparse_operation_none) ? (B + tid) : (B + ldb * tid)));

    for(I at = csx_ptr[tid] - csx_base; at < csx_ptr[tid + 1] - csx_base; ++at)
    {
        I        ind = csx_ind[at] - csx_base;
        const T* y
            = (row_oriented)
                  ? ((orderB == rocsparse_order_column)
                         ? ((transB == rocsparse_operation_none) ? (B + ldb * ind) : (B + ind))
                         : ((transB == rocsparse_operation_none) ? (B + ind) : (B + ldb * ind)))
                  : ((orderA == rocsparse_order_column)
                         ? ((transA == rocsparse_operation_none) ? (A + ind) : (A + lda * ind))
                         : ((transA == rocsparse_operation_none) ? (A + lda * ind) : (A + ind)));

        T sum = static_cast<T>(0);
        for(J k = local_thread_index; k < K; k += NTHREADS_PER_DOTPRODUCT)
        {
            sum += x[k * xinc] * y[k * yinc];
        }
        s[local_seq_index][local_thread_index] = sum;
        __syncthreads();

#pragma unroll
        for(int ipow2_ = 2; ipow2_ <= NTHREADS_PER_DOTPRODUCT; ipow2_ *= 2)
        {
            if(local_thread_index < NTHREADS_PER_DOTPRODUCT / ipow2_)
            {
                s[local_seq_index][local_thread_index]
                    += s[local_seq_index][local_thread_index + NTHREADS_PER_DOTPRODUCT / ipow2_];
            }
            __syncthreads();
        }

        if(local_thread_index == 0)
        {
            csx_val[at] = csx_val[at] * beta + alpha * s[local_seq_index][0];
        }
    }
}
