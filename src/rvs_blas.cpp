/********************************************************************************
 *
 * Copyright (c) 2018 ROCm Developer Tools
 *
 * MIT LICENSE:
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "include/rvs_blas.h"

#include <ctime>
#include <iostream>
#include <cmath>
#define RANDOM_CT               320000
#define RANDOM_DIV_CT           0.1234
#define PI                      3.14159265
#define RAD_FIVE                0.0872
#define RAD_NTY                 1.5708
/**
 * @brief class constructor
 * @param _gpu_device_index the gpu that will run the GEMM
 * @param _m matrix size
 * @param _n matrix size
 * @param _k matrix size
 */
rvs_blas::rvs_blas(int _gpu_device_index, int _m, int _n, int _k, int transA, int transB, 
                    float alpha , float beta, rocblas_int lda, rocblas_int ldb, rocblas_int ldc, std::string ops_type) : gpu_device_index(_gpu_device_index),
                             m(_m), n(_n), k(_k){
    is_handle_init = false;
    is_error = false;
    da = db = dc = NULL;
    ha = hb = hc = nullptr;

    if(transA == 0) {
         transa = rocblas_operation_none;
    }else{
         transa = rocblas_operation_transpose;
    }

    if(transB == 0) {
         transb = rocblas_operation_none;
    }else{
         transb = rocblas_operation_transpose;
    }

    if(ops_type == "hgemm") {
        auto    A_row = transA == rocblas_operation_none ? m : k;
        auto    A_col = transA == rocblas_operation_none ? k : m;
        auto    B_row = transB == rocblas_operation_none ? k : n;
        auto    B_col = transB == rocblas_operation_none ? n : k;

        size_a = size_t(lda) * size_t(A_col);
        size_b = size_t(ldb) * size_t(B_col);
        size_c = size_t(ldc) * size_t(n);
        size_d = size_t(ldc) * size_t(n);
    }else{
      size_a = k * m;
      size_b = k * n;
      size_c = n * m;
    }

    if (alocate_host_matrix_mem()) {
        if (!init_gpu_device())
            is_error = true;
    } else {
        is_error = true;
    }

    //setting alpha and beta val
    blas_alpha_val = alpha;
    blas_beta_val = beta;

    //setting lda offsets 
    //Leading data offsets
    if(lda == 0 || ldb == 0 || ldc == 0) {
       blas_lda_offset = m;
       blas_ldb_offset = n;
       blas_ldc_offset = k;
    }else{
       blas_lda_offset = lda;
       blas_ldb_offset = ldb;
       blas_ldc_offset = ldc;
    }
}

/**
 * @brief class destructor
 */
rvs_blas::~rvs_blas() {
    release_host_matrix_mem();
    release_gpu_matrix_mem();
}

/**
 * @brief selects GPU device, allocates GPU memory, creates a rocBlas
 * handle and get a reference to the rocBlas's stream
 * @return true if everything went fine, otherwise false
 */
bool rvs_blas::init_gpu_device(void) {
    // select GPU device & allocate memory
    if (hipSetDevice(gpu_device_index) != hipSuccess) {
        // cannot select the given GPU device
        return false;
    } else {
       //ROCBLAS Initialize
        rocblas_initialize();

        if (!allocate_gpu_matrix_mem())
            return false;
        if (rocblas_create_handle(&blas_handle) == rocblas_status_success) {
            is_handle_init = true;
            if (rocblas_get_stream(blas_handle, &hip_stream)
                 != rocblas_status_success)
                return false;
        } else {
            return false;
        }
    }
    return true;
}

/**
 * @brief copy data matrix from host to gpu
 * @return true if everything went fine, otherwise false
 */
bool rvs_blas::copy_data_to_gpu(std::string ops_type) {

      if(ops_type == "sgemm") {

            if (da) {
                 if (hipMemcpy(da, ha, sizeof(float) * size_a, hipMemcpyHostToDevice)
                                 != hipSuccess) {
                      is_error = true;
                      return false;
                  }
            }

            if (db) {
                if (hipMemcpy(db, hb, sizeof(float) * size_b, hipMemcpyHostToDevice)
                                != hipSuccess) {
                      is_error = true;
                      return false;
                  }
            }

            if (dc) {
                  if (hipMemcpy(dc, hc, sizeof(float) * size_c, hipMemcpyHostToDevice)
                          != hipSuccess) {
                       is_error = true;
                       return false;
                  }
             }
      }


      if(ops_type == "dgemm") {

            if (ddbla) {
                  if (hipMemcpy(ddbla, hdbla, sizeof(double) * size_a, hipMemcpyHostToDevice)
                                     != hipSuccess) {
                        is_error = true;
                        return false;
                  }
            }

            if (ddblb) {
                  if (hipMemcpy(ddblb, hdblb, sizeof(double) * size_b, hipMemcpyHostToDevice)
                                    != hipSuccess) {
                        is_error = true;
                        return false;
                  }
            }

            if (ddblc) {
                  if (hipMemcpy(ddblc, hdblc, sizeof(double) * size_c, hipMemcpyHostToDevice)
                                  != hipSuccess) {
                       is_error = true;
                       return false;
                  }
            }

      }

      if(ops_type == "hgemm") {

            if (dhlfa) {
                  if (hipMemcpy(dhlfa, hhlfa, sizeof(rocblas_half) * size_a, hipMemcpyHostToDevice)
                                     != hipSuccess) {
                        is_error = true;
                        return false;
                  }
            }

            if (dhlfb) {
                  if (hipMemcpy(dhlfb, hhlfb, sizeof(rocblas_half) * size_b, hipMemcpyHostToDevice)
                                    != hipSuccess) {
                        is_error = true;
                        return false;
                  }
            }

            if (dhlfc) {
                  if (hipMemcpy(dhlfc, hhlfc, sizeof(rocblas_half) * size_c, hipMemcpyHostToDevice)
                                  != hipSuccess) {
                       is_error = true;
                       return false;
                  }
            }

      }


    is_error = false;
    return true;
}

/**
 * @brief allocates memory (for matrix multiplication) on the selected GPU
 * @return true if everything went fine, otherwise false
 */
bool rvs_blas::allocate_gpu_matrix_mem(void) {
    if (hipMalloc(&da, size_a * sizeof(float)) != hipSuccess)
        return false;
    if (hipMalloc(&db, size_b * sizeof(float)) != hipSuccess)
        return false;
    if (hipMalloc(&dc, size_c * sizeof(float)) != hipSuccess)
        return false;

    if (hipMalloc(&ddbla, size_a * sizeof(double)) != hipSuccess)
        return false;
    if (hipMalloc(&ddblb, size_b * sizeof(double)) != hipSuccess)
        return false;
    if (hipMalloc(&ddblc, size_c * sizeof(double)) != hipSuccess)
        return false;

    if (hipMalloc(&dhlfa, size_a * sizeof(rocblas_half)) != hipSuccess)
        return false;
    if (hipMalloc(&dhlfb, size_b * sizeof(rocblas_half)) != hipSuccess)
        return false;
    if (hipMalloc(&dhlfc, size_c * sizeof(rocblas_half)) != hipSuccess)
        return false;
    if (hipMalloc(&dhlfd, size_d * sizeof(rocblas_half)) != hipSuccess)
        return false;

    return true;
}

/**
 * @brief gets time
 */
double rvs_blas::get_time_us(void)
{
    hipDeviceSynchronize();
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000 * 1000) + tv.tv_usec;
};
 

/**
 * @brief releases GPU mem & destroys the rocBlas handle
 */
void rvs_blas::release_gpu_matrix_mem(void) {
    if (da)
        hipFree(da);
    if (db)
        hipFree(db);
    if (dc)
        hipFree(dc);

    if (ddbla)
        hipFree(ddbla);
    if (ddblb)
        hipFree(ddblb);
    if (ddblc)
        hipFree(ddblc);

    if (dhlfa)
        hipFree(dhlfa);
    if (dhlfb)
        hipFree(dhlfb);
    if (dhlfc)
        hipFree(dhlfc);
    if (dhlfd)
        hipFree(dhlfd);

    if (is_handle_init)
        rocblas_destroy_handle(blas_handle);
}

/**
 * @brief allocate host matrix memory
 * @return true if everything went fine, otherwise false
 */
bool rvs_blas::alocate_host_matrix_mem(void) {

    try {
        ha = new float[size_a];
        hb = new float[size_b];
        hc = new float[size_c];

        hdbla = new double[size_a];
        hdblb = new double[size_b];
        hdblc = new double[size_c];

        hhlfa = new rocblas_half[size_a];
        hhlfb = new rocblas_half[size_b];
        hhlfc = new rocblas_half[size_c];

        return true;
    } catch (std::bad_alloc&) {
        return false;
    }
}

/**
 * @brief releases the host matrix memory
 */
void rvs_blas::release_host_matrix_mem(void) {
    if (ha)
        delete []ha;
    if (hb)
        delete []hb;
    if (hc)
        delete []hc;

    if (hdbla)
        delete []hdbla;
    if (hdblb)
        delete []hdblb;
    if (hdblc)
        delete []hdblc;

    if (hhlfa)
        delete []hhlfa;
    if (hhlfb)
        delete []hhlfb;
    if (hhlfc)
        delete []hhlfc;
}

/**
 * @brief checks whether the matrix multiplication completed
 * @return true if GPU finished with matrix multiplication, otherwise false
 */
bool rvs_blas::is_gemm_op_complete(void) {
    if (is_error)
        return true;  // avoid blocking the calling thread
    if (hipStreamQuery(hip_stream) != hipSuccess)
        return false;
    return true;
}

/**
 * @brief performs the SGEMM matrix multiplication
 * @return true if GPU was able to enqueue the GEMM operation, otherwise false
 */
bool rvs_blas::run_blass_gemm(std::string ops_type) {

    if (!is_error) {
      
        if(ops_type == "sgemm") {

                 float alpha = blas_alpha_val, beta = blas_beta_val;
                 
                 if (rocblas_sgemm(blas_handle, transa, transb,
                         rvs_blas::m, rvs_blas::n, rvs_blas::k,
                         &alpha, da, blas_lda_offset,
                         db, blas_ldb_offset, &beta,
                         dc, blas_ldc_offset) != rocblas_status_success) {
                    is_error = true;  // GPU cannot enqueue the gemm
                    return false;
                 } else {
                      return true;
                 }
        }

       if(ops_type == "dgemm") {

                  double alpha = blas_alpha_val, beta = blas_beta_val;

                  if (rocblas_dgemm(blas_handle, transa, transb,
                          rvs_blas::m, rvs_blas::n, rvs_blas::k,
                          &alpha, ddbla, blas_lda_offset,
                          ddblb, blas_ldb_offset, &beta,
                          ddblc, blas_ldc_offset) != rocblas_status_success) {
                       is_error = true;  // GPU cannot enqueue the gemm
                       return false;
                  } else {
                       return true;
                  }
       }

      if(ops_type == "hgemm") {
                  rocblas_float alpha;
                  rocblas_float beta;
                  rocblas_datatype a_type = rocblas_datatype_f16_r;
                  rocblas_datatype b_type = rocblas_datatype_f16_r;
                  rocblas_datatype c_type = rocblas_datatype_f16_r;
                  rocblas_datatype d_type = rocblas_datatype_f16_r;
                  rocblas_datatype compute_type = rocblas_datatype_f32_r;
                  rocblas_gemm_algo algo = static_cast<rocblas_gemm_algo>(0);
                  int sol_index = 0;
                  int flags = 10;

                  alpha = blas_alpha_val;
                  beta  = blas_beta_val;

#if 0
		  std::cout << "\n M size : " << rvs_blas::m;
		  std::cout << "\n N size : " << rvs_blas::n;
		  std::cout << "\n K size : " << rvs_blas::k;
		  std::cout << "\n Alpha : " << alpha;
		  std::cout << "\n Beta : " << beta;
		  std::cout << "\n LDA : " << blas_lda_offset;
		  std::cout << "\n LDB : " << blas_ldb_offset;
		  std::cout << "\n LDC : " << blas_ldc_offset;
#endif

                  if (rocblas_gemm_ex(blas_handle, transa, transb,
                          rvs_blas::m, rvs_blas::n, rvs_blas::k,
                          &alpha, dhlfa, a_type, blas_lda_offset,
                          dhlfb, b_type, blas_ldb_offset, &beta,
                          dhlfc, c_type, blas_ldc_offset,
                          dhlfc, c_type, blas_ldc_offset,
                          compute_type, algo, sol_index, flags) != rocblas_status_success) {
                       is_error = true;  // GPU cannot enqueue the gemm
                       std::cout << "\n Error in Hgemm " << "\n";
                       return false;
                  } else {
                       return true;
                  }
       }

    } else {
        return false;
    }

    return true;
}

/**
 * @brief generate matrix random data
 * it should be called before rocBlas GEMM
 */
void rvs_blas::generate_random_matrix_data(void) {
    int i;
    if (!is_error) {
        uint64_t nextr = time(NULL);

        //SGEMM stuff
        for (i = 0; i < size_a; ++i)
            ha[i] = fast_pseudo_rand(&nextr);

        for (i = 0; i < size_b; ++i)
            hb[i] = fast_pseudo_rand(&nextr);

        for (int i = 0; i < size_c; ++i)
            hc[i] = fast_pseudo_rand(&nextr);

        //DGEMM stuff
        for (i = 0; i < size_a; ++i)
            hdbla[i] = (double)fast_pseudo_rand(&nextr);

        for (i = 0; i < size_b; ++i)
            hdblb[i] = (double)fast_pseudo_rand(&nextr);

        for (int i = 0; i < size_c; ++i)
            hdblc[i] = (double)fast_pseudo_rand(&nextr);

        for (i = 0; i < size_a; ++i)
            hhlfa[i].data = (uint16_t)fast_pseudo_rand(&nextr);

        for (i = 0; i < size_b; ++i)
            hhlfb[i].data = (uint16_t)fast_pseudo_rand(&nextr);

        for (int i = 0; i < size_c; ++i)
            hhlfc[i].data = (uint16_t)fast_pseudo_rand(&nextr);
    }
}



/**
 * @brief fast pseudo random generator 
 * @return floating point random number
 */
float rvs_blas::fast_pseudo_rand(u_long *nextr) {
    *nextr = *nextr * 1103515245 + 12345;
    return static_cast<float>(static_cast<uint32_t>
                    ((*nextr / 65536) % RANDOM_CT)) / RANDOM_DIV_CT;
}

/**
 * @brief generate trig matrix data
 * it should be called before rocBlas GEMM
 */
void rvs_blas::generate_trig_matrix_data(void) {
    int i;
    if (!is_error) {
        uint64_t nextr = time(NULL);
        
        //SGEMM stuff
        for (i = 0; i < size_a; ++i)
            ha[i] = trig_rand(&nextr, true);

        for (i = 0; i < size_b; ++i)
            hb[i] = trig_rand(&nextr, false);

        for (int i = 0; i < size_c; ++i)
            hc[i] = trig_rand(&nextr, true);

        //DGEMM stuff
        for (i = 0; i < size_a; ++i)
            hdbla[i] = (double)trig_rand(&nextr, true);

        for (i = 0; i < size_b; ++i)
            hdblb[i] = (double)trig_rand(&nextr, false);

        for (int i = 0; i < size_c; ++i)
            hdblc[i] = (double)trig_rand(&nextr, true);
        
        // HGEMM stuff
        for (i = 0; i < size_a; ++i)
            hhlfa[i].data = (uint16_t)trig_rand(&nextr, true);

        for (i = 0; i < size_b; ++i)
            hhlfb[i].data = (uint16_t)trig_rand(&nextr, false);

        for (int i = 0; i < size_c; ++i)
            hhlfc[i].data = (uint16_t)trig_rand(&nextr, true);
    }

}

/**
 * @brief genertaes a trig value based on long input
 * @return floating point random trigonometric value	
 */
float rvs_blas::trig_rand(u_long *nextr, bool sin) {
    *nextr = *nextr * 1103515245 + 12345;
    float temp = static_cast<float>(static_cast<uint32_t>
                    ((*nextr / 65536) % RANDOM_CT)) / RANDOM_DIV_CT;
    // ranging between 5 degree(0.0872 radians) and 90 degree(1.5708)
    float rads = RAD_FIVE + temp * ( RAD_NTY - RAD_FIVE ); 
    if(sin)
        return std::sin(rads);
    return cos(rads);
}

