/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_csr2csr_compress.hpp"
#include "definitions.h"
#include "utility.h"

#include "csr2csr_compress_device.h"
#include <rocprim/rocprim.hpp>

template <typename T>
rocsparse_status rocsparse_csr2csr_compress_template(rocsparse_handle          handle,
                                                     rocsparse_int             m,
                                                     rocsparse_int             n,
                                                     const rocsparse_mat_descr descr_A,
                                                     const T*                  csr_val_A,
                                                     const rocsparse_int*      csr_row_ptr_A,
                                                     const rocsparse_int*      csr_col_ind_A,
                                                     rocsparse_int             nnz_A,
                                                     const rocsparse_int*      nnz_per_row,
                                                     T*                        csr_val_C,
                                                     rocsparse_int*            csr_row_ptr_C,
                                                     rocsparse_int*            csr_col_ind_C,
                                                     T                         tol)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xcsr2csr_compress"),
              m,
              n,
              descr_A,
              (const void*&)csr_val_A,
              (const void*&)csr_row_ptr_A,
              (const void*&)csr_col_ind_A,
              nnz_A,
              (const void*&)nnz_per_row,
              (const void*&)csr_val_C,
              (const void*&)csr_row_ptr_C,
              (const void*&)csr_col_ind_C,
              tol);

    log_bench(
        handle, "./rocsparse-bench -f csr2csr_compress -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    // Check matrix descriptor
    if(descr_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check matrix sorting mode
    if(descr_A->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0 || nnz_A < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check tolerance
    if(std::real(tol) < std::real(static_cast<T>(0)))
    {
        return rocsparse_status_invalid_value;
    }

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(csr_row_ptr_A == nullptr || csr_row_ptr_C == nullptr || nnz_per_row == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val_A == nullptr && csr_col_ind_A != nullptr)
       || (csr_val_A != nullptr && csr_col_ind_A == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val_C == nullptr && csr_col_ind_C != nullptr)
       || (csr_val_C != nullptr && csr_col_ind_C == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_A != 0 && (csr_val_A == nullptr && csr_col_ind_A == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Compute required temporary storage buffer size
    size_t nwarps                   = (nnz_A - 1) / handle->wavefront_size + 1;
    size_t temp_storage_size_bytes1 = sizeof(int) * (nwarps / 256 + 1) * 256;

    auto   op = rocprim::plus<rocsparse_int>();
    size_t temp_storage_size_bytes2;
    RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(nullptr,
                                                temp_storage_size_bytes2,
                                                (rocsparse_int*)nullptr,
                                                (rocsparse_int*)nullptr,
                                                m + 1,
                                                op,
                                                stream));
    temp_storage_size_bytes2 = ((temp_storage_size_bytes2 - 1) / 256 + 1) * 256;

    size_t temp_storage_size_bytes3;
    RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(nullptr,
                                                temp_storage_size_bytes3,
                                                (rocsparse_int*)nullptr,
                                                (rocsparse_int*)nullptr,
                                                nwarps + 1,
                                                op,
                                                stream));
    temp_storage_size_bytes3 = ((temp_storage_size_bytes3 - 1) / 256 + 1) * 256;

    size_t temp_storage_size_bytes
        = temp_storage_size_bytes1 + temp_storage_size_bytes2 + temp_storage_size_bytes3;

    bool  temp_alloc       = false;
    void* temp_storage_ptr = nullptr;
    if(handle->buffer_size >= temp_storage_size_bytes)
    {
        temp_storage_ptr = handle->buffer;
        temp_alloc       = false;
    }
    else
    {
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(&temp_storage_ptr, temp_storage_size_bytes, handle->stream));
        temp_alloc = true;
    }

    char* ptr        = reinterpret_cast<char*>(temp_storage_ptr);
    int*  warp_start = reinterpret_cast<int*>(ptr);
    ptr += temp_storage_size_bytes1;
    void* temp_storage_buffer2 = ptr;
    ptr += temp_storage_size_bytes2;
    void* temp_storage_buffer3 = ptr;
    ptr += temp_storage_size_bytes3;

    // Copy nnz_per_row to csr_row_ptr_C array
    hipLaunchKernelGGL((fill_row_ptr_device<1024>),
                       dim3((m - 1) / 1024 + 1),
                       dim3(1024),
                       0,
                       stream,
                       m,
                       descr_A->base,
                       nnz_per_row,
                       csr_row_ptr_C);

    // Perform inclusive scan on csr row pointer array
    RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(temp_storage_buffer2,
                                                temp_storage_size_bytes2,
                                                csr_row_ptr_C,
                                                csr_row_ptr_C,
                                                m + 1,
                                                op,
                                                stream));

    if(csr_val_C == nullptr && csr_col_ind_C == nullptr)
    {
        rocsparse_int start = 0;
        rocsparse_int end   = 0;

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &end, &csr_row_ptr_C[m], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&start,
                                           &csr_row_ptr_C[0],
                                           sizeof(rocsparse_int),
                                           hipMemcpyDeviceToHost,
                                           handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        rocsparse_int nnz_C = (end - start);

        if(nnz_C != 0)
        {
            return rocsparse_status_invalid_pointer;
        }
    }

#define LOOPS 2
    if(handle->wavefront_size == 32)
    {
        hipLaunchKernelGGL((csr2csr_compress_fill_warp_start_device<256, 32, LOOPS>),
                           dim3((nnz_A - 1) / (256 * LOOPS) + 1),
                           dim3(256),
                           0,
                           stream,
                           nnz_A,
                           csr_val_A,
                           warp_start,
                           tol);
    }
    else if(handle->wavefront_size == 64)
    {
        hipLaunchKernelGGL((csr2csr_compress_fill_warp_start_device<256, 64, LOOPS>),
                           dim3((nnz_A - 1) / (256 * LOOPS) + 1),
                           dim3(256),
                           0,
                           stream,
                           nnz_A,
                           csr_val_A,
                           warp_start,
                           tol);
    }

    // Perform inclusive scan on warp start array
    RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(temp_storage_buffer3,
                                                temp_storage_size_bytes3,
                                                warp_start,
                                                warp_start,
                                                nwarps + 1,
                                                op,
                                                stream));

    if(handle->wavefront_size == 32)
    {
        hipLaunchKernelGGL((csr2csr_compress_use_warp_start_device<256, 32, LOOPS>),
                           dim3((nnz_A - 1) / (256 * LOOPS) + 1),
                           dim3(256),
                           0,
                           stream,
                           nnz_A,
                           descr_A->base,
                           csr_val_A,
                           csr_col_ind_A,
                           descr_A->base,
                           csr_val_C,
                           csr_col_ind_C,
                           warp_start,
                           tol);
    }
    else if(handle->wavefront_size == 64)
    {
        hipLaunchKernelGGL((csr2csr_compress_use_warp_start_device<256, 64, LOOPS>),
                           dim3((nnz_A - 1) / (256 * LOOPS) + 1),
                           dim3(256),
                           0,
                           stream,
                           nnz_A,
                           descr_A->base,
                           csr_val_A,
                           csr_col_ind_A,
                           descr_A->base,
                           csr_val_C,
                           csr_col_ind_C,
                           warp_start,
                           tol);
    }
#undef LOOPS

    if(temp_alloc)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(temp_storage_ptr, handle->stream));
    }

    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define C_IMPL(NAME, TYPE)                                                    \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,        \
                                     rocsparse_int             m,             \
                                     rocsparse_int             n,             \
                                     const rocsparse_mat_descr descr_A,       \
                                     const TYPE*               csr_val_A,     \
                                     const rocsparse_int*      csr_row_ptr_A, \
                                     const rocsparse_int*      csr_col_ind_A, \
                                     rocsparse_int             nnz_A,         \
                                     const rocsparse_int*      nnz_per_row,   \
                                     TYPE*                     csr_val_C,     \
                                     rocsparse_int*            csr_row_ptr_C, \
                                     rocsparse_int*            csr_col_ind_C, \
                                     TYPE                      tol)           \
    {                                                                         \
        return rocsparse_csr2csr_compress_template(handle,                    \
                                                   m,                         \
                                                   n,                         \
                                                   descr_A,                   \
                                                   csr_val_A,                 \
                                                   csr_row_ptr_A,             \
                                                   csr_col_ind_A,             \
                                                   nnz_A,                     \
                                                   nnz_per_row,               \
                                                   csr_val_C,                 \
                                                   csr_row_ptr_C,             \
                                                   csr_col_ind_C,             \
                                                   tol);                      \
    }

C_IMPL(rocsparse_scsr2csr_compress, float);
C_IMPL(rocsparse_dcsr2csr_compress, double);
C_IMPL(rocsparse_ccsr2csr_compress, rocsparse_float_complex);
C_IMPL(rocsparse_zcsr2csr_compress, rocsparse_double_complex);
#undef C_IMPL
