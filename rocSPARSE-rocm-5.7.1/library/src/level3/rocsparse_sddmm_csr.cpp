/* ************************************************************************
 * Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_sddmm_csx_kernel.hpp"

template <typename I, typename J, typename T>
struct rocsparse_sddmm_st<rocsparse_format_csr, rocsparse_sddmm_alg_default, I, J, T>
{
    /*获取所需的缓冲区大小。返回0表示不需要额外的缓冲区*/
    static rocsparse_status buffer_size(rocsparse_handle     handle,
                                        rocsparse_operation  trans_A,
                                        rocsparse_operation  trans_B,
                                        rocsparse_order      order_A,
                                        rocsparse_order      order_B,
                                        J                    m,
                                        J                    n,
                                        J                    k,
                                        I                    nnz,
                                        const T*             alpha,
                                        const T*             A_val,
                                        J                    A_ld,
                                        const T*             B_val,
                                        J                    B_ld,
                                        const T*             beta,
                                        const I*             C_row_data,
                                        const J*             C_col_data,
                                        T*                   C_val_data,
                                        rocsparse_index_base C_base,
                                        rocsparse_sddmm_alg  alg,
                                        size_t*              buffer_size)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    /*预处理SDDMM操作，但这里啥也没干*/
    static rocsparse_status preprocess(rocsparse_handle     handle,
                                       rocsparse_operation  trans_A,
                                       rocsparse_operation  trans_B,
                                       rocsparse_order      order_A,
                                       rocsparse_order      order_B,
                                       J                    m,
                                       J                    n,
                                       J                    k,
                                       I                    nnz,
                                       const T*             alpha,
                                       const T*             A_val,
                                       J                    A_ld,
                                       const T*             B_val,
                                       J                    B_ld,
                                       const T*             beta,
                                       const I*             C_row_data,
                                       const J*             C_col_data,
                                       T*                   C_val_data,
                                       rocsparse_index_base C_base,
                                       rocsparse_sddmm_alg  alg,
                                       void*                buffer)
    {
        return rocsparse_status_success;
    }

    /*核心函数：执行SDDMM操作的计算*/
    static rocsparse_status compute(rocsparse_handle     handle,
                                    rocsparse_operation  trans_A,
                                    rocsparse_operation  trans_B,
                                    rocsparse_order      order_A,
                                    rocsparse_order      order_B,
                                    J                    m,
                                    J                    n,
                                    J                    k,
                                    I                    nnz,
                                    const T*             alpha,
                                    const T*             A_val,
                                    J                    A_ld,
                                    const T*             B_val,
                                    J                    B_ld,
                                    const T*             beta,
                                    const I*             C_row_data,
                                    const J*             C_col_data,
                                    T*                   C_val_data,
                                    rocsparse_index_base C_base,
                                    rocsparse_sddmm_alg  alg,
                                    void*                buffer)
    {
        static constexpr int NB = 512; // NB：线程块大小，即，每个线程块包含多少个线程
/*
定义了一个名为 HLAUNCH 的宏，用于启动一个 GPU kernel
(NT_) 表示宏的参数，即传入的参数 NT_，表示线程块内的线程数量
后面的部分是宏定义中的代码块：
    计算了一个名为 num_blocks_x 的整数变量的值，确定需要启动多少个线程块来处理数据：m表示矩阵维度，NB表示线程块大小，NT_表示一个线程块内的线程数量。
    变量blocks表示启动 kernel 时的线程块数量。
    变量threads表示每个线程块内的线程数量
    hipLaunchKernelGGL()启动了 GPU 上的一个 kernel 函数，即：启动sddmm_csx_kernel这个核函数。
    （NB线程块大小，NT_线程块内线程数量，blocks启动的线程块数量，threads每个线程块内的线程数量，额外参数0通常用于设置共享内存的大小，handle->stream是GPU上下文的流对象，表示GPU核函数将在哪个流上执行 ）
    （接下来的参数是 GPU Kernel 函数的参数）

    *(const T*)alpha 表示解引用指针 alpha，以获取指针指向的实际值，这里的alpha应该为指针。
*/
#define HLAUNCH(NT_)                                                                  \
    int64_t num_blocks_x = (m - 1) / (NB / NT_) + 1;                                  \
    dim3    blocks(num_blocks_x);                                                     \
    dim3    threads(NB);                                                              \
    hipLaunchKernelGGL((sddmm_csx_kernel<NB, NT_, rocsparse_direction_row, I, J, T>), \
                       blocks,                                                        \
                       threads,                                                       \
                       0,                                                             \
                       handle->stream,                                                \
                       trans_A,                                                       \
                       trans_B,                                                       \
                       order_A,                                                       \
                       order_B,                                                       \
                       m,                                                             \
                       n,                                                             \
                       k,                                                             \
                       nnz,                                                           \
                       *(const T*)alpha,                                              \
                       A_val,                                                         \
                       A_ld,                                                          \
                       B_val,                                                         \
                       B_ld,                                                          \
                       *(const T*)beta,                                               \
                       (T*)C_val_data,                                                \
                       (const I*)C_row_data,                                          \
                       (const J*)C_col_data,                                          \
                       C_base,                                                        \
                       (T*)buffer)

#define DLAUNCH(NT_)                                                                  \
    int64_t num_blocks_x = (m - 1) / (NB / NT_) + 1;                                  \
    dim3    blocks(num_blocks_x);                                                     \
    dim3    threads(NB);                                                              \
    hipLaunchKernelGGL((sddmm_csx_kernel<NB, NT_, rocsparse_direction_row, I, J, T>), \
                       blocks,                                                        \
                       threads,                                                       \
                       0,                                                             \
                       handle->stream,                                                \
                       trans_A,                                                       \
                       trans_B,                                                       \
                       order_A,                                                       \
                       order_B,                                                       \
                       m,                                                             \
                       n,                                                             \
                       k,                                                             \
                       nnz,                                                           \
                       alpha,                                                         \
                       A_val,                                                         \
                       A_ld,                                                          \
                       B_val,                                                         \
                       B_ld,                                                          \
                       beta,                                                          \
                       (T*)C_val_data,                                                \
                       (const I*)C_row_data,                                          \
                       (const J*)C_col_data,                                          \
                       C_base,                                                        \
                       (T*)buffer)

        // 这个if检查 handle 对象的 pointer_mode 是否为 rocsparse_pointer_mode_host，即是否使用主机指针模式。
        // 主机指针模式表示数据在主机内存上，并且 alpha 和 beta 参数是主机上的指针。
        if(handle->pointer_mode == rocsparse_pointer_mode_host)
        {
            // 如果alpha为0，beta为1，则无需操作
            if(*alpha == static_cast<T>(0) && *beta == static_cast<T>(1))
            {
                return rocsparse_status_success;
            }
            // 根据 k 的值，选择不同的线程块大小 NT_，然后调用 HLAUNCH(NT_) 或 DLAUNCH(NT_) 来启动 GPU Kernel 来执行计算。
            if(k > 4)
            {
                HLAUNCH(8);
            }
            else if(k > 2)
            {
                HLAUNCH(4);
            }
            else if(k > 1)
            {
                HLAUNCH(2);
            }
            else
            {
                HLAUNCH(1);
            }
        }
        // 设备指针模式，即数据在 GPU 上，并且 alpha 和 beta 参数是 GPU 上的指针。
        else
        {
            if(k > 4)
            {
                DLAUNCH(8);
            }
            else if(k > 2)
            {
                DLAUNCH(4);
            }
            else if(k > 1)
            {
                DLAUNCH(2);
            }
            else
            {
                DLAUNCH(1);
            }
        }
        return rocsparse_status_success;
    }
};

// 模板实例化
/* 
rocsparse_format_csr：表示使用的稀疏矩阵格式为 CSR (Compressed Sparse Row) 格式。
rocsparse_sddmm_alg_default：表示使用的 SDDMM (Sparse-Dense Matrix Multiplication) 算法为默认算法。
rocsparse_double_complex：双精度复数
*/
template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int32_t,
                                   int32_t,
                                   float>;
template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int32_t,
                                   int32_t,
                                   double>;
template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int32_t,
                                   int32_t,
                                   rocsparse_float_complex>;
template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int32_t,
                                   int32_t,
                                   rocsparse_double_complex>;

template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int32_t,
                                   float>;
template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int32_t,
                                   double>;
template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int32_t,
                                   rocsparse_float_complex>;
template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int32_t,
                                   rocsparse_double_complex>;

template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int64_t,
                                   float>;
template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int64_t,
                                   double>;
template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int64_t,
                                   rocsparse_float_complex>;
template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int64_t,
                                   rocsparse_double_complex>;
