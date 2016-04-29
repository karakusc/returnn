
#include <string.h>
#include <assert.h>

#define ARRAY_LEN(x) (sizeof(x) / sizeof(x[0]))

#if CUDA

// Defined here: https://github.com/Theano/Theano/blob/master/theano/sandbox/cuda/cuda_ndarray.cuh
// See also: https://github.com/Theano/Theano/blob/master/theano/sandbox/cuda/cuda_ndarray.cu
#define Ndarray CudaNdarray
#define Ndarray_DEV_DATA CudaNdarray_DEV_DATA
#define Ndarray_HOST_DIMS CudaNdarray_HOST_DIMS
#define Ndarray_DIMS Ndarray_HOST_DIMS
#define Ndarray_NDIM(x) (x->nd)
#define Ndarray_DIM_Type int
#define Ndarray_SIZE CudaNdarray_SIZE
// PyObject *CudaNdarray_NewDims(int nd, const inttype * dims), uninitialized
#define Ndarray_NewDims CudaNdarray_NewDims
// PyObject * CudaNdarray_Copy(const CudaNdarray * self);
#define Ndarray_Copy CudaNdarray_Copy
#define Ndarray_memcpy(y, x, size) (cudaMemcpy(y, x, size, cudaMemcpyDeviceToDevice))
/*
    // via: http://docs.nvidia.com/cuda/cublas/
    // matrices are in column-major form
    cublasStatus_t cublasSgemm(cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const float *alpha, const float *A, int lda,
        const float *B, int ldb, const float *beta,
        float *C, int ldc);
*/
#define _cublasTranspose(t) \
	((t == 'T') ? CUBLAS_OP_T : \
	(t == 'C') ? CUBLAS_OP_C : \
	(t == 'N') ? CUBLAS_OP_N : cublasOperation_t('E'))
#define Ndarray_sgemm( \
	transpose_A, transpose_B, \
	m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) \
	(_cudaHandleError(cublasSgemm(handle, \
	_cublasTranspose(transpose_A), \
	_cublasTranspose(transpose_B), \
	m, n, k, alpha, A, lda, B, ldb, beta, C, ldc), \
	__FILE__, __LINE__ ))

#define DIM_GRID 128
#define DIM_BLOCK 512

#define DEF_KERNEL __global__
#define start_dev_kernel(kernel, args) \
	(kernel<<<DIM_GRID,DIM_BLOCK>>>  args);

static const char *_cudaGetErrorEnum(cublasStatus_t error) {
	switch (error) {
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";

	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";

	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";

	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";

	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";

	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";

	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";

	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
	}

	return "<unknown>";
}

static void _cudaHandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

static void _cudaHandleError(cublasStatus_t status, const char *file, int line) {
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("%s in %s at line %d\n", _cudaGetErrorEnum(status), file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR(err) (_cudaHandleError( err, __FILE__, __LINE__ ))

#else   // not CUDA

// Numpy, see: http://docs.scipy.org/doc/numpy/reference/c-api.array.html
// And: http://deeplearning.net/software/theano/extending/extending_theano_c.html
#define Ndarray PyArrayObject
#define Ndarray_DEV_DATA(x) ((float*) PyArray_DATA(x))
#define Ndarray_HOST_DIMS PyArray_DIMS
#define Ndarray_DIMS Ndarray_HOST_DIMS
#define Ndarray_NDIM PyArray_NDIM
#define Ndarray_DIM_Type npy_intp
#define Ndarray_SIZE PyArray_SIZE
#define Ndarray_NewDims(nd, dims) (PyArray_SimpleNew(nd, dims, NPY_FLOAT32))
#define Ndarray_Copy(x) (PyArray_FromArray(x, NULL, 0))
#define Ndarray_memcpy(y, x, size) (memcpy(y, x, size))
/*
    // matrices are in column-major form
	int sgemm_(char *transa, char *transb,
		integer *m, integer *n, integer *k,
		real *alpha, real *a, integer *lda,
		real *b, integer *ldb, real *beta,
		real *c, integer *ldc);
*/
#define Ndarray_sgemm(\
	transpose_A, transpose_B, \
	m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) \
	{ \
		char transa = transpose_A, transb = transpose_B; \
		int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc; \
		sgemm_(&transa, &transb, \
			&m_, &n_, &k_, alpha, A, &lda_, B, &ldb_, beta, C, &ldc_); \
	}

#define DEF_KERNEL
#define start_dev_kernel(kernel, args) \
	{ for(_KernelLoop loop; !loop.finished(); loop.next()) { kernel args; } }

struct vec3 {
	int x; int y; int z;
	void reset() { x = y = z = 0; }
};

vec3 gridDim;
vec3 blockDim;
vec3 threadIdx;
vec3 blockIdx;

struct _KernelLoop {
	_KernelLoop() {
		// When we can choose whatever we want here, this loops becomes trivial,
		// there will only be one iteration.
		gridDim.reset(); gridDim.x = 1;
		blockDim.reset(); blockDim.x = 1;
		threadIdx.reset();
		blockIdx.reset();
	}
	bool finished() {
		// TODO: Also block idx but doesn't matter with the constants above.
		// TODO: Also y/z but doesn't matter with the constants above.
		return threadIdx.x >= blockDim.x;
	}
	void next() {
		// TODO: Also blockIdx and y/z, but doesn't matter with the constants above.
		threadIdx.x++;
	}
};

#endif

Ndarray* Ndarray_uninitialized_like(Ndarray* a) {
	const Ndarray_DIM_Type* dim = Ndarray_HOST_DIMS(a);
	Ndarray* res = (Ndarray*) Ndarray_NewDims(Ndarray_NDIM(a), (Ndarray_DIM_Type*) dim);
	return res;
}

//if nd is 2 then assume a weight matrix and just return beginning of data
//else nd should be 3 and we pick the x part
float* data_ptr(Ndarray* a, int x) {
	assert(Ndarray_NDIM(a) == 2 || Ndarray_NDIM(a) == 3);
	if(Ndarray_NDIM(a) == 2)
		return Ndarray_DEV_DATA(a);
	else {
		const Ndarray_DIM_Type* dims = Ndarray_HOST_DIMS(a);
		return Ndarray_DEV_DATA(a) + x * dims[1] * dims[2];
	}
}

const float* data_ptr(const Ndarray* a, int x) {
	return data_ptr((Ndarray*) a, x);
}

void lastTwoDims(const Ndarray* a, int out[2]) {
	const Ndarray_DIM_Type* dims = Ndarray_HOST_DIMS((Ndarray*) a);
	assert(Ndarray_NDIM(a) >= 2);
	out[0] = dims[Ndarray_NDIM(a) - 2];
	out[1] = dims[Ndarray_NDIM(a) - 1];
}

int lastTwoDimsStride(const Ndarray * a) {
	int dims[2];
	lastTwoDims(a, dims);
	return dims[0] * dims[1];
}

DEF_KERNEL void lstm_kernel(float* data, const float* old_state, bool old_state_strided,
                            float* output, float* state_out, int n_cells, int n_batch, const float* i) {
	//layout:
	//data[0*n_cells..1*n_cells-1] : input gate
	//data[1*n_cells..2*n_cells-1] : forget gate
	//data[2*n_cells..3*n_cells-1] : output gate
	//data[3*n_cells..4*n_cells-1] : cell state
	//output[0*n_cells..1*n_cells-1]: cell output
	//repeated for every mini-batch

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	while (idx < n_cells * n_batch) {
	    int batch_idx = idx / n_cells;
		int start = batch_idx * 4 * n_cells + idx % n_cells;
		float i_batch = i[batch_idx];

		//input, forget and output gates
		float inpGate = 1.f / (1.f + expf(-data[start + n_cells]));
		float fgtGate = 1.f / (1.f + expf(-data[start + 2 * n_cells]));
		float outGate = 1.f / (1.f + expf(-data[start + 3 * n_cells]));
		float state = inpGate * tanhf(data[start]);
		float old_state_batch = old_state_strided ? old_state[start] : old_state[idx];

		state += fgtGate * old_state_batch;
		state = state * i_batch + old_state_batch * (1.f - i_batch);

		//cell output
		output[idx] = outGate * tanhf(state) * i_batch;

		data[start] = state;
		data[start + n_cells] = inpGate;
		data[start + 2 * n_cells] = fgtGate;
		data[start + 3 * n_cells] = outGate;
		if(state_out)
		    state_out[idx] = state;

		idx += gridDim.x * blockDim.x;
	}
}

DEF_KERNEL void lstm_bwd_kernel(
        float* delta, float* epsilon, const float* next_epsilon, const float* old_state,
        bool old_state_strided, const float* Y, int n_cells, int n_batch, const float* i) {
	//layout:
	//delta[0*n_cells..1*n_cells-1] : input gate
	//delta[1*n_cells..2*n_cells-1] : forget gate
	//delta[2*n_cells..3*n_cells-1] : output gate
	//delta[3*n_cells..4*n_cells-1] : cell state
	//epsilon[0*n_cells..1*n_cells-1]: cell output derivative (later overwritten, see below)
	//next_epsilon[0*n_cells..1*n_cells-1]: cell state derivative * forget_gate (of next timestep)
	//repeated for every mini-batch

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	while (idx < n_cells * n_batch) {
		int batch_idx = idx / n_cells;
		int batch_offset = batch_idx * 4 * n_cells;
		int cell_offset = idx % n_cells;
		int start = batch_offset + cell_offset;
		float i_batch = i[batch_idx];

		float inpGate = delta[start + n_cells];
		float fgtGate = delta[start + 2 * n_cells];
		float outGate = delta[start + 3 * n_cells];
		float oldState = old_state_strided ? old_state[start] : old_state[idx];
		float state = delta[start];
		float eps = epsilon[idx];

		//avoid division by 0 (TODO: check if this is needed)
		float gc = 0.f; //g(c(t))
		float gzc = 0.f; //g(z_c(t))
		if (outGate != 0)
			gc = Y[idx] / outGate;
		if (inpGate != 0)
			gzc = (state - fgtGate * oldState) / inpGate;

		//delta_output
		delta[start + 3 * n_cells] = outGate * (1.f - outGate) * gc * eps * i_batch;

		//epsilon_c
		float epsilon_c = (1.f - (gc * gc)) * outGate * eps;
		epsilon_c += next_epsilon[idx];
		epsilon[idx] = epsilon_c * fgtGate * i_batch + next_epsilon[idx] * (1.f - i_batch);

		//delta_cell
		delta[start] = inpGate * (1.f - (gzc * gzc)) * epsilon_c * i_batch;

		//delta_forget
		delta[start + 2 * n_cells] = fgtGate * (1.f - fgtGate) * oldState * epsilon_c * i_batch;

		//delta_input
		delta[start + n_cells] = inpGate * (1.f - inpGate) * gzc * epsilon_c * i_batch;

		idx += gridDim.x * blockDim.x;
	}
}

//C[x] += A[x]*B[x]
//(if not 4-dimensional, then indexing [x] is ignored (e.g. for weight matrices))
void affine_y_x(int x_A, Ndarray* A, int x_B, Ndarray* B,
	            int x_C, /*out*/Ndarray* C, bool transpose_A = false, bool transpose_B = false) {
	const float* data_A = data_ptr(A, x_A);
	const float* data_B = data_ptr(B, x_B);
	float* data_C = data_ptr(C, x_C);
	int A_dim[2], B_dim[2];
	lastTwoDims(A, A_dim);
	lastTwoDims(B, B_dim);

	int ldB = B_dim[1];
	int ldA = A_dim[1];
	char transA = transpose_A ? 'T' : 'N';
	char transB = transpose_B ? 'T' : 'N';
	if (transpose_A)
		std::swap(A_dim[0], A_dim[1]);
	if (transpose_B)
		std::swap(B_dim[0], B_dim[1]);

	const float alpha = 1;
	const float beta = 1;

	Ndarray_sgemm(transB, transA, B_dim[1], A_dim[0], A_dim[1], &alpha, data_B, ldB,
		data_A, ldA, &beta, data_C, B_dim[1]);
}

//offset is used for x time-shift between A and B
//if offset == 1, then we will calculate A[0..end-1] * B[1..end]
void affine_global(Ndarray* A, Ndarray* B, /*out*/Ndarray* C,
                   bool transpose_A = false, bool transpose_B = false, int offset = 0, float beta = 1.0) {
	float* data_C = Ndarray_DEV_DATA(C);
	int A_dim[2], B_dim[2];
	lastTwoDims(A, A_dim);
	lastTwoDims(B, B_dim);
	int shiftA = A_dim[1] * A_dim[0];
	int shiftB = B_dim[1] * B_dim[0];
	A_dim[0] = Ndarray_SIZE(A) / A_dim[1] - offset * A_dim[0];
	B_dim[0] = Ndarray_SIZE(B) / B_dim[1] - offset * A_dim[0];
	const float * data_A = Ndarray_DEV_DATA(A);
	const float * data_B = Ndarray_DEV_DATA(B) + offset * shiftB;

	int ldB = B_dim[1];
	int ldA = A_dim[1];
	char transA = transpose_A ? 'T' : 'N';
	char transB = transpose_B ? 'T' : 'N';
	if (transpose_A)
		std::swap(A_dim[0], A_dim[1]);
	if (transpose_B)
		std::swap(B_dim[0], B_dim[1]);

	const float alpha = 1;
	Ndarray_sgemm(transB, transA, B_dim[1], A_dim[0], A_dim[1], &alpha, data_B, ldB,
		data_A, ldA, &beta, data_C, B_dim[1]);
}
