#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <iostream>
#include <cblas.h>  // Include the C interface for the BLAS library
#include <lapacke.h> // for matrix inverses
// for patch-setup
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include <chrono>

namespace py = pybind11;

void n_part( const py::array_t<int64_t> block_sizes, 
             const py::array_t<int64_t> offset_bs,
             const py::array_t<int64_t> offset_nnz, 
             const py::array_t<double>  A_inv_flat, 
             const py::array_t<double>  r_gs, 
                   py::array_t<double>  x_gs,
             const py::array_t<double>  scale_glob,
             const double               omega
            ) {
    auto block_sizes_ptr = block_sizes.data();
    auto A_inv_flat_ptr  =  A_inv_flat.data();
    auto offset_bs_ptr   =   offset_bs.data();
    auto offset_nnz_ptr  =  offset_nnz.data();
    auto r_gs_ptr        =        r_gs.data();
    auto scale_glob_ptr  =  scale_glob.data();
    auto x_gs_ptr        =        x_gs.mutable_data();

    int num_block_sizes = block_sizes.size();

    for (int p = 0; p < num_block_sizes; ++p) {
        const int bs                  = block_sizes_ptr[p];
        // constant pointer and constant data
        const double* const A_inv     = A_inv_flat_ptr + offset_nnz_ptr[p];
        const double* const r_t       =       r_gs_ptr + offset_bs_ptr[p];
        double* const x_t             =       x_gs_ptr + offset_bs_ptr[p];
        const double* const scale_vec = scale_glob_ptr + offset_bs_ptr[p];

        cblas_dgemv(CblasRowMajor,
                    CblasNoTrans, 
                    bs, bs, 
                    1.0, 
                    A_inv, 
                    bs, 
                    r_t, 
                    1, 
                    0.0, 
                    x_t, 
                    1);
        // scale the patch-wise solution vector by partion of unity and omega
        for (int jj = 0; jj < bs; ++jj)
            x_t[jj] = omega * x_t[jj] * scale_vec[jj];
    }
}

void n_part_batch(
                  const py::array_t<int64_t> patch_size, 
                  const py::array_t<int64_t> patch_count, 
                  const py::array_t<int64_t> l_slice, 
                  const py::array_t<int64_t> q_slice, 
                  const py::array_t<double>  r_gs, 
                  const py::array_t<double>  A_inv_flat, 
                        py::array_t<double>  x_gs,
                  const py::array_t<double>  scale_glob,
                  const double               omega
                  ) {
    
    auto patch_size_ptr  =  patch_size.data();
    auto patch_count_ptr = patch_count.data();
    auto l_slice_ptr     =     l_slice.data();
    auto q_slice_ptr     =     q_slice.data();
    auto r_gs_ptr        =        r_gs.data();
    auto A_inv_flat_ptr  =  A_inv_flat.data();
    auto scale_glob_ptr  =  scale_glob.data();
    auto x_gs_ptr        =        x_gs.mutable_data();

    const unsigned int num_batches = patch_size.size();
    for (unsigned int i = 0; i < num_batches; ++i) {
        const int patch_size = patch_size_ptr[i];
        const int count      = patch_count_ptr[i];
        //std::cout<< "patch_size=" << patch_size << ", count=" << count <<"\n";
        double *r_batch     = const_cast<double*>(r_gs_ptr)       + l_slice_ptr[i];
        double *x_batch     = const_cast<double*>(x_gs_ptr)       + l_slice_ptr[i];
        double *scale_batch = const_cast<double*>(scale_glob_ptr) + l_slice_ptr[i];
        double *Ainv_batch  = const_cast<double*>(A_inv_flat_ptr) + q_slice_ptr[i];

        for (int j = 0; j < count; ++j) {
            const long offset = j * patch_size;
            double *Ainv_mat  = Ainv_batch + offset * patch_size;
            double *r_vec     =    r_batch + offset;
            double *out_r_vec =    x_batch + offset;
            double *scale_vec =scale_batch + offset;

            // Perform matrix-vector multiplication using the BLAS dgemv function
            // Y = alpha*Ax + beta*y
            cblas_dgemv(CblasRowMajor, // storage order of the matrix A
                        CblasNoTrans,  // if the matrix A should be transposed 
                        patch_size,    //  number of rows in A 
                        patch_size,    //  number of cols in A
                        1.0,           // alpha 
                        Ainv_mat,      // A
                        patch_size,    // leading dimension of A
                        r_vec,         // pointer to the first element of x
                        1,             // stride between elements of the x
                        0,             // beta
                        out_r_vec,     // y
                        1              // stride between elements of y
                        );
            
            // scale the patch-wise solution vector by partion of unity
            // and omega
            for (int jj = 0; jj < patch_size; ++jj) {
                out_r_vec[jj] = omega*out_r_vec[jj] * scale_vec[jj];
            }
        }
    }
}

//Global variables to store loop timings
std::chrono::duration<double> loop1_time(0);
std::chrono::duration<double> loop2_time(0);
std::chrono::duration<double> loop3_time(0);

void n_part_factorized(
                  const py::array_t<int64_t> patch_size, 
                  const py::array_t<int64_t> patch_count, 
                  const py::array_t<int64_t> l_slice, 
                  const py::array_t<int64_t> q_slice, 
                  const py::array_t<int64_t> p_slice, 
                  const py::array_t<double>  M_inv_flat, 
                  const py::array_t<double>  B_hat_flat, 
                  const py::array_t<double>  S_inv_flat, 
                  const py::array_t<double>  U_hat_flat, 
                  const py::array_t<double>  ru_gs, 
                        py::array_t<double>  u_gs,
                  const py::array_t<double>  rp,
                        py::array_t<double>  dp,
                  const int dim
                  ) {
    
    auto patch_size_ptr  =  patch_size.data();
    auto patch_count_ptr = patch_count.data();
    auto l_slice_ptr     =     l_slice.data();
    auto q_slice_ptr     =     q_slice.data();
    auto p_slice_ptr     =     p_slice.data();
    auto ru_gs_ptr       =       ru_gs.data();
    auto rp_ptr          =          rp.data();
    auto M_inv_flat_ptr  =  M_inv_flat.data();
    auto B_hat_flat_ptr  =  B_hat_flat.data();
    auto S_inv_flat_ptr  =  S_inv_flat.data();
    auto U_hat_flat_ptr  =  U_hat_flat.data();
    auto u_gs_ptr        =        u_gs.mutable_data();
    auto   dp_ptr        =          dp.mutable_data();

    const unsigned int num_batches = patch_size.size();
    for (unsigned int i = 0; i < num_batches; ++i) {
        const unsigned int patch_size = patch_size_ptr[i];
        const int count      = patch_count_ptr[i];
        double *ru_batch     = const_cast<double*>(ru_gs_ptr)    + l_slice_ptr[i];
        double *u_batch      = const_cast<double*>(u_gs_ptr)     + l_slice_ptr[i];
        double *Minv_batch   = const_cast<double*>(M_inv_flat_ptr) + q_slice_ptr[i];

        for (int j = 0; j < count*dim; ++j) {
            const unsigned int offset = j * patch_size;
            double *Minv_mat    =  Minv_batch + offset * patch_size;
            double *ru_vec      =    ru_batch + offset;
            double *out         =     u_batch + offset;

            // Perform matrix-vector multiplication using the BLAS dgemv function
            // Y = alpha*Ax + beta*y
            cblas_dgemv(CblasRowMajor, // storage order of the matrix A
                        CblasNoTrans,  // if the matrix A should be transposed
                        patch_size,    //  number of rows in A
                        patch_size,    //  number of cols in A
                        1.0,           // alpha
                        Minv_mat,      // A
                        patch_size,    // leading dimension of A
                        ru_vec,        // pointer to the first element of x
                        1,             // stride between elements of the x
                        0,             // beta
                        out      ,     // y
                        1              // stride between elements of y
                        );
//            cblas_dsymv(CblasRowMajor, // storage order of the matrix A
//                        CblasUpper,     // specify if the upper or lower part of the symmetric matrix A should be used
//                        patch_size,     // number of rows in A
//                        1.0,            // alpha
//                        Minv_mat,       // A
//                        patch_size,     // leading dimension of A
//                        ru_vec,         // pointer to the first element of x
//                        1,              // stride between elements of x
//                        0,              // beta
//                        out,            // y
//                        1               // stride between elements of y
//                        );
        }

        double* Bhat_batch = const_cast<double*>(B_hat_flat_ptr) + l_slice_ptr[i];
        double* Sinv_batch = const_cast<double*>(S_inv_flat_ptr) + p_slice_ptr[i];
        double* Uhat_batch = const_cast<double*>(U_hat_flat_ptr) + l_slice_ptr[i];
        double*    p_batch =                             dp_ptr  + p_slice_ptr[i];
        double*   rp_batch = const_cast<double*>(        rp_ptr) + p_slice_ptr[i];
        for (int row = 0; row < count; ++row) {
            const long offset = row * patch_size * dim;
            double *Bhat_row    =  Bhat_batch + offset;
            double *Uhat_row    =  Uhat_batch + offset;
            double *u_vec       =     u_batch + offset;
            double *ru_row      =    ru_batch + offset;

            p_batch[row] = cblas_ddot(patch_size * dim, 
                                       Bhat_row, 1, 
                                         ru_row, 1);
            p_batch[row] += Sinv_batch[row]*rp_batch[row];
        
            cblas_daxpy(patch_size*dim, 
                        p_batch[row], 
                        Uhat_row, 1, 
                        u_vec, 1);
        }
    }
}

std::vector<long int> precompute_A_offsets(
    const py::array_t<int>& rowptr_B,
    const py::array_t<int>& pdof_order,
    const int pdof_per_patch
) {
    std::vector<long int> A_offsets(pdof_order.shape(0) + 1, 0);
    for (int p = 0; p < pdof_order.shape(0); ++p) {
        int p_idx        = pdof_order.at(p);
        int n_vertices   = rowptr_B.at(p_idx + 1) - rowptr_B.at(p_idx);
        int n_cols       = n_vertices + pdof_per_patch;
        A_offsets[p + 1] = A_offsets[p] + n_cols * n_cols;
    }
    return A_offsets;
}

// error checks
void check_lapack_info(const std::string& function_name, int info) {
    if (info < 0) {
        std::cerr << "Error: The " << -info << "-th argument of " 
		  << function_name << " had an illegal value." 
		  << std::endl;
        exit(EXIT_FAILURE);
    } else if (info > 0) {
        if (function_name == "dgetrf_") {
            std::cerr << "Error: U(" << info << "," << info 
		      << ") in the LU decomposition is exactly zero." 
		      << std::endl;
        } else if (function_name == "dgetri_") {
            std::cerr << "Error: U(" << info << "," << info 
		    << ") in the LU decomposition is exactly zero;"
		    << " the matrix is singular and its inverse could not be computed." 
		    << std::endl;
        } else {
            std::cerr << "Error: " << function_name 
		      << " returned a positive info value: " 
		      << info << std::endl;
        }
        exit(EXIT_FAILURE);
    }
}

/* Setup the patch data structure
 *
 *
*/

// global to local v_id mapping
static inline
int binary_search(const int* arr, int size, int target) {
    int left = 0, right = size - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target)
            return mid;
        if (arr[mid] < target)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return -1;
}
static inline
void extract_vector_laplacian_matrix(int n_vertices, int n_cols,
                                     const int* rowptr_M_ptr,
                                     const int* col_indices_M_ptr,
                                     const double* data_M_ptr,
                                     const int* v_ids_ptr,
                                     double* Aloc_ptr) {
    for (int i = 0; i < n_vertices; ++i) {
        int row = v_ids_ptr[i];
        for (int j = rowptr_M_ptr[row]; j < rowptr_M_ptr[row + 1]; ++j) {
            int col      = col_indices_M_ptr[j];
            double value = data_M_ptr[j];
            int col_idx  = binary_search(v_ids_ptr, n_vertices, col);
            if (col_idx >= 0 && col_idx <= i) {
                Aloc_ptr[i * n_cols + col_idx] = Aloc_ptr[col_idx * n_cols + i] = value;
            }
        }
    }
}
static inline
void extract_divergence_operator(int p_idx, int n_vertices, int n_cols,
                                 const int* rowptr_B_ptr,
                                 const double* data_B_ptr,
                                 double* Aloc_ptr)
{
    for (int col_Mloc = 0; col_Mloc < n_vertices; ++col_Mloc) {
        double value = data_B_ptr[rowptr_B_ptr[p_idx] + col_Mloc];
        Aloc_ptr[n_vertices * n_cols + col_Mloc] = value;
        Aloc_ptr[col_Mloc * n_cols + n_vertices] = value;
    }
}

static inline
void extract_divergence_operator_dg(int p, int n_vertices, int n_cols,
                                   const int* pdof_order_ptr,
                                   const int* rowptr_B_ptr,
                                   const int* col_indices_B_ptr,
                                   const double* data_B_ptr,
                                   double* Aloc_ptr,
                                   const std::vector<int>& v_ids)
{
    for (int j = 0; j < 3; ++j) {
        const int p_idx = pdof_order_ptr[p * 3 + j];
        int n_vertices_B = rowptr_B_ptr[p_idx + 1] - rowptr_B_ptr[p_idx];
        const int* v_ids_ptr = col_indices_B_ptr + rowptr_B_ptr[p_idx];

        for (int col_Mloc = 0; col_Mloc < n_vertices_B; ++col_Mloc) {
            double value = data_B_ptr[rowptr_B_ptr[p_idx] + col_Mloc];
            int col = v_ids_ptr[col_Mloc];
            int col_idx = binary_search(v_ids.data(), n_vertices, col);
            Aloc_ptr[(n_vertices + j) * n_cols + col_idx] = value;
            Aloc_ptr[col_idx * n_cols + n_vertices + j] = value;
        }
    }
}

static inline
void check_lapack_info(const char* function_name, int info) {
    if (info < 0) {
        std::cerr << function_name
                  << " had an illegal argument: "
                  << -info << std::endl;
    } else if (info > 0) {
        std::cerr << function_name
                  << " failed: U(" << info << ","
                  << info << ") is exactly zero" << std::endl;
    }
}

static inline
void compute_inverse(int n_cols, double* Aloc_ptr, int* ipiv_ptr, double*& work, int& l_work)
{
    if (n_cols == 1)
    {
        Aloc_ptr[0] = 1.0 / Aloc_ptr[0];
        return;
    }

    int lda = n_cols; // Leading dimension of the matrix A
    int info;
    dgetrf_(&n_cols, &n_cols, Aloc_ptr, &lda, ipiv_ptr, &info);
    check_lapack_info("dgetrf_", info);

    // find optimal workspace size
    int lwork = -1;
    double work_query;
    dgetri_(&n_cols, Aloc_ptr, &lda, ipiv_ptr, &work_query, &lwork, &info);
    lwork = static_cast<int>(work_query); // Set the workspace size to the optimal value
    if (lwork > l_work){
        free(work);
        work = static_cast<double*>(malloc(lwork * sizeof(double)));
        l_work = lwork;
        std::cout << "realloc=" << lwork << "\n";
    }
    // Compute the inverse of A
    dgetri_(&n_cols, Aloc_ptr, &lda, ipiv_ptr, work, &lwork, &info);
    check_lapack_info("dgetri_", info);
}


// Patch Setup
void th_patch_setup(
    const py::array_t<int>& rowptr_M,
    const py::array_t<int>& col_indices_M,
    const py::array_t<double>& data_M,
    const py::array_t<int>& rowptr_B,
    const py::array_t<int>& col_indices_B,
    const py::array_t<double>& data_B,
    const py::array_t<int>& pdof_order,
    py::array_t<double>& _A,
    const int max_block_size
) {

    // Getting the input data
    const int* rowptr_M_ptr = rowptr_M.data();
    const int* col_indices_M_ptr = col_indices_M.data();
    const double* data_M_ptr = data_M.data();
    const int* pdof_order_ptr = pdof_order.data();
    double* _A_ptr = _A.mutable_data();

    const int* rowptr_B_ptr = rowptr_B.data();
    const int* col_indices_B_ptr = col_indices_B.data();
    const double* data_B_ptr = data_B.data();

    // Precompute A offsets (for parallelization)
    std::vector<long int> A_offsets = precompute_A_offsets(rowptr_B, pdof_order, 1);
    for (int p = 0; p < pdof_order.shape(0); ++p) {
        long int _A_offset   = A_offsets[p];
        const int p_idx = pdof_order_ptr[p];

        int n_vertices  = rowptr_B_ptr[p_idx + 1] - rowptr_B_ptr[p_idx];
        int n_cols      = n_vertices + 1;
        double* Aloc_ptr = _A_ptr + _A_offset;

        // unique velocity nodes
        const int* v_ids_ptr = col_indices_B_ptr + rowptr_B_ptr[p_idx];

        extract_vector_laplacian_matrix(n_vertices, n_cols,
                                        rowptr_M_ptr, col_indices_M_ptr,
                                        data_M_ptr, v_ids_ptr, Aloc_ptr);

        extract_divergence_operator(p, n_vertices, n_cols,
                                    rowptr_B_ptr, data_B_ptr, Aloc_ptr);
    } // p_idx

    // Compute inverse via  LU factorization
    int* ipiv_ptr = (int*) malloc(max_block_size * sizeof(int));
    int l_work    = max_block_size*max_block_size*10;
    double* work  = (double*) malloc(l_work * sizeof(double));

    for (int p = 0; p < pdof_order.shape(0); ++p) {
        long int _A_offset   = A_offsets[p];
        const int p_idx = pdof_order_ptr[p];
        int n_vertices  = rowptr_B_ptr[p_idx + 1] - rowptr_B_ptr[p_idx];
        int n_cols      = n_vertices+1;

        double* Aloc_ptr = _A_ptr+_A_offset;
        compute_inverse(n_cols, Aloc_ptr, ipiv_ptr, work, l_work);
    }
    free(work);
    free(ipiv_ptr);
}

/* CBLAS calls have overhead for small matrices/vectorts.
 * We use our own implementation
 * for small matrices.
 * At this point the the switching threshold is set to a small number.
 * This should be optimized. (TODO)
 */
inline
double my_cblas_ddot(const int n,
                     const double *x, const int incx,
                     const double *y, const int incy) {
    /* */
    if (n < 150)
    {
        double result = 0.0;
        for (int i = 0; i < n; ++i) {
            result += x[i] * y[i];
        }
        return result;
    } else  {
        return cblas_ddot(n, x, incx, y, incy);
    }
}

inline
void my_cblas_dgemv(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA,
                    const int M, const int N,
                    const double alpha, const double *A, const int lda,
                    const double *X, const int incX,
                    const double beta, double *Y,
                    const int incY) {
    /* */
    if (M < 100){
        for (int row = 0; row < M; ++row) {
            double temp = 0.0;
            for (int col = 0; col < N; ++col) {
                temp += alpha * A[row * M + col] * X[col];
            }
            Y[row] = temp +  beta * Y[row];
        }
    } else {
        cblas_dgemv(
                    layout, TransA,
                    M, //  number of rows in A
                    N, //  number of cols in A
                    alpha, // alpha
                    A, // A
                    lda, // leading dimension of A
                    X,  // pointer to the first element of x
                    incX, // stride between elements of the x
                    beta, // beta
                    Y, // y
                    incY // stride between elements of y
                    );
    }
}

void th_bf_patch_setup(
    const py::array_t<int>& rowptr_M,
    const py::array_t<int>& col_indices_M,
    const py::array_t<double>& data_M,
    const py::array_t<int>& rowptr_B,
    const py::array_t<int>& col_indices_B,
    const py::array_t<double>& data_B,
    const py::array_t<int>& Mloc_x_sizes,
    py::array_t<double>& _Minv,
    py::array_t<double>& _Uhat,
    py::array_t<double>& _Sinv,
    const int dim,
    const int max_block_size
) {

    // Getting the input data
    const int*      rowptr_M_ptr = rowptr_M.data();
    const int* col_indices_M_ptr = col_indices_M.data();
    const double*     data_M_ptr = data_M.data();

    const int*      rowptr_B_ptr = rowptr_B.data();
    const int* col_indices_B_ptr = col_indices_B.data();
    const double*     data_B_ptr = data_B.data();

    const int* Mloc_x_sizes_ptr  = Mloc_x_sizes.data();

    double* Minv_ptr = _Minv.mutable_data();
    double* Uhat_ptr = _Uhat.mutable_data();
    double* Sinv_ptr = _Sinv.mutable_data();

    size_t patch_num =  static_cast<size_t>(Mloc_x_sizes.size() / dim);
    // Compute inverse via  LU factorization
    int* ipiv_ptr = (int*) malloc(max_block_size * sizeof(int));
    int l_work    = max_block_size*max_block_size*20;
    double* work  = (double*) malloc(l_work * sizeof(double));

    long int Bloc_offset = 0;
    long int M_offset = 0;
    for (size_t p_idx = 0; p_idx < patch_num; ++p_idx) {
        //long int M_offset   = M_offsets[p_idx];

        int n_vertices      = rowptr_B_ptr[p_idx + 1] - rowptr_B_ptr[p_idx];
        int n_cols          = n_vertices;
        double* Minv        =   Minv_ptr +    M_offset;
        const double* Bloc  = data_B_ptr + Bloc_offset;
        double* Uhat        =   Uhat_ptr + Bloc_offset;

        // unique velocity nodes
        const int* v_ids    = col_indices_B_ptr + rowptr_B_ptr[p_idx];

         long M_size  = 0;
         long M_size2 = 0;
         for (int j = 0; j < dim; ++j) {
            int bs = Mloc_x_sizes_ptr[p_idx * dim + j];

            extract_vector_laplacian_matrix(bs, bs,
                                            rowptr_M_ptr, col_indices_M_ptr, data_M_ptr,
                                            v_ids + M_size,
                                            Minv  + M_size2);
            compute_inverse(bs, Minv + M_size2, ipiv_ptr, work, l_work);
            cblas_dgemv(
			            CblasRowMajor,
                        CblasNoTrans,
                        bs, //  number of rows in A
                        bs, //  number of cols in A
                        -1.0, // alpha
                        &Minv[M_size2], // A
                        bs, // leading dimension of A
                        &Bloc[M_size],  // pointer to the first element of x
                        1, // stride between elements of the x
                        0, // beta
                        &Uhat[M_size], // y
                        1 // stride between elements of y
                        );

            M_size  += bs;
            M_size2 += bs*bs;
         } // j

         Sinv_ptr[p_idx] = 1.0/my_cblas_ddot(M_size,
                                              Bloc, 1,
                                              Uhat, 1);
         Bloc_offset += n_cols;
         M_offset += M_size2;
    } // p_idx

    free(work);
    free(ipiv_ptr);
}


void sv_patch_setup(
    const py::array_t<int>& rowptr_M, 
    const py::array_t<int>& col_indices_M,
    const py::array_t<double>& data_M,
    const py::array_t<int>& rowptr_B, 
    const py::array_t<int>& col_indices_B, 
    const py::array_t<double>& data_B,
    const py::array_t<int>& pdof_order,
    const int pcells,
    py::array_t<double>& _A,
    const int max_block_size
) {
    // assumes 2D triangular mesh
    // Getting the input data
    const int* rowptr_M_ptr = rowptr_M.data();
    const int* col_indices_M_ptr = col_indices_M.data();
    const double* data_M_ptr = data_M.data();
    const int* pdof_order_ptr = pdof_order.data();
    double* _A_ptr = _A.mutable_data();

    const int* rowptr_B_ptr = rowptr_B.data();
    const int* col_indices_B_ptr = col_indices_B.data();
    const double* data_B_ptr = data_B.data();

    int* ipiv_ptr = (int*) malloc(max_block_size * sizeof(int));
    int l_work    = max_block_size*max_block_size*10;
    double* work  = (double*) malloc(l_work * sizeof(double));

    long int _A_offset = 0;
    for (int p = 0; p < pcells; ++p) {
        // unique velocity nodes
        std::vector<int> v_ids;
        for (int j = 0; j < 3; ++j) {
            const int p_idx = pdof_order_ptr[p * 3 + j];
            int n_vertices  = rowptr_B_ptr[p_idx + 1] - rowptr_B_ptr[p_idx];

            const int* v_ids_ptr = col_indices_B_ptr + rowptr_B_ptr[p_idx];
            v_ids.insert(v_ids.end(), v_ids_ptr, v_ids_ptr + n_vertices);
        }
        std::sort(v_ids.begin(), v_ids.end());
        v_ids.erase(std::unique(v_ids.begin(), v_ids.end()), v_ids.end());

        int n_vertices = static_cast<int>(v_ids.size());
        int n_cols = n_vertices + 3;

        double* Aloc_ptr = _A_ptr + _A_offset;
        extract_vector_laplacian_matrix(n_vertices, n_cols,
                                        rowptr_M_ptr, col_indices_M_ptr,
                                        data_M_ptr, v_ids.data(), Aloc_ptr);

        extract_divergence_operator_dg(p, n_vertices, n_cols,
                                       pdof_order_ptr,
                                       rowptr_B_ptr,
                                       col_indices_B_ptr,
                                       data_B_ptr,
                                       Aloc_ptr,
                                       v_ids);

        compute_inverse(n_cols, Aloc_ptr, ipiv_ptr, work, l_work);

        _A_offset += n_cols * n_cols;
    }
    free(work);
    free(ipiv_ptr);
}


void block_fact_patch_setup(
    const py::array_t<int>& rowptr_M,
    const py::array_t<int>& col_indices_M,
    const py::array_t<double>& data_M,
    const py::array_t<int>& rowptr_B,
    const py::array_t<int>& col_indices_B,
    //const py::array_t<double>& data_B,
    const py::array_t<int>& Ps_ordered,
    const py::array_t<int>& Mloc_x_sizes_padded,
    py::array_t<double>& M_inv_padded,
   const int max_block_size,
   const int dim
) {
    const int* rowptr_M_ptr      = rowptr_M.data();
    const int* col_indices_M_ptr = col_indices_M.data();
    const double* data_M_ptr     = data_M.data();
    const int* rowptr_B_ptr      = rowptr_B.data();
    const int* col_indices_B_ptr = col_indices_B.data();
    //const double* data_B_ptr     = data_B.data();
    const int* Ps_ordered_ptr    = Ps_ordered.data();

    const int* Mloc_x_sizes_padded_ptr = Mloc_x_sizes_padded.data();

    double* M_inv_padded_ptr = M_inv_padded.mutable_data();

    int* ipiv_ptr    = (int*) malloc(max_block_size * sizeof(int));
    int l_work       = max_block_size*max_block_size*10;
    double* work_ptr = (double*) malloc(l_work * sizeof(double));

    long M_inv_nnz_offset_padded = 0;
    //long B_nnz_offset_padded = 0;
    //long B_nnz_offset = 0;

    size_t size = Ps_ordered.size();
    for (size_t i = 0; i < size; ++i)
    {
        int p_idx          = Ps_ordered_ptr[i];
        int Ux_size_padded = Mloc_x_sizes_padded_ptr[i];


        int n_vertices  = rowptr_B_ptr[p_idx + 1] - rowptr_B_ptr[p_idx];
        int n_cols      = n_vertices;
        // unique velocity nodes
        const int* v_ids_ptr = col_indices_B_ptr + rowptr_B_ptr[p_idx];
        double *  M_inv_padded_loc_ptr = M_inv_padded_ptr + M_inv_nnz_offset_padded;

        extract_vector_laplacian_matrix(n_vertices, n_cols,
                                        rowptr_M_ptr, col_indices_M_ptr, data_M_ptr,
                                        v_ids_ptr, M_inv_padded_loc_ptr);

        M_inv_nnz_offset_padded += dim*(n_vertices)*(Ux_size_padded);
    }

    free(work_ptr);
    free(ipiv_ptr);
}


// block-factotized Vanka (simplified)
void n_part_bs(const py::array_t<int>& Mloc_x_sizes,
               const py::array_t<double>& M_inv,
               const py::array_t<double>& r,
               py::array_t<double>& x,
               const int dim,
               const int u_dofs
               ) {

    const int* Mloc_x_sizes_ptr = Mloc_x_sizes.data();
    const double* M_inv_ptr     = M_inv.data();

    const double* r_ptr     = r.data();
    double* x_ptr           = x.mutable_data();

    const double* ru_ptr = r_ptr;
    const double* rp_ptr = r_ptr+u_dofs;
    double* du_ptr = x_ptr;
    double* dp_ptr = x_ptr+u_dofs;

    long U_offset = 0;
    long M_offset = 0;

    size_t patch_num =  static_cast<size_t>(Mloc_x_sizes.size() / dim);

    for (size_t i = 0; i < patch_num; ++i) {

        long Mx_offset = 0;
        long Mx_offset2 = 0;
        long M_size = 0;
        long M_size2 = 0;

        //#pragma unroll
        for (int j = 0; j < dim; ++j) {
            M_size += Mloc_x_sizes_ptr[i * dim + j];
            M_size2 += Mloc_x_sizes_ptr[i * dim + j] * Mloc_x_sizes_ptr[i * dim + j];
        }

        double dot_product = my_cblas_ddot(M_size,
                                           du_ptr + U_offset, -1, // dummy
                                           ru_ptr + U_offset, -1  // dummy
                                           );
        dp_ptr[i] *= (dot_product + rp_ptr[i]);

        for (int j = 0; j < dim; ++j) {
            int bs = Mloc_x_sizes_ptr[i * dim + j];
            int bs2 = bs * bs;

            my_cblas_dgemv(
                        CblasRowMajor,
                        CblasNoTrans,
                        bs, //  number of rows in A
                        bs, //  number of cols in A
                        1.0, // alpha
                        &M_inv_ptr[M_offset + Mx_offset2], // A
                        bs, // leading dimension of A
                        &ru_ptr[U_offset + Mx_offset],  // pointer to the first element of x
                        1, // stride between elements of the x
                        dp_ptr[i], // beta
                        &du_ptr[U_offset + Mx_offset], // y
                        1 // stride between elements of y
                        );

            Mx_offset += bs;
            Mx_offset2 += bs2;
        }
        U_offset += M_size;
        M_offset += M_size2;
    }
}



PYBIND11_MODULE(patch_mult, m) {
    m.doc() = "Matrix operations using pybind11";
    m.def("n_part"      ,      &n_part, "n_part patch-vector multiplication");
    m.def("n_part_bs", &n_part_bs, "(simplified) Block Factorized patch-vector multiplication");
    m.def("n_part_batch",      &n_part_batch, "Batched patch-vector multiplication");  
    
    m.def("sv_patch_setup", &sv_patch_setup, "Scott-Vogelius algebraic patch setup all");
    m.def("th_bf_patch_setup", &th_bf_patch_setup, "Taylor-Hood block-factorized algebraic patch setup");
    m.def("th_patch_setup", &th_patch_setup, "Taylor-Hood algebraic patch setup all");
    // not used atm. 
    m.def("n_part_factorized", &n_part_factorized, "Factorized patch-vector multiplication");
    m.def("block_fact_patch_setup", &block_fact_patch_setup, "Taylor-Hood algebraic block-factorized patch setup");
}

