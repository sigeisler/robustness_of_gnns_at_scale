#include <torch/extension.h>

#include <ATen/cuda/CUDAUtils.h>
#include <ATen/SparseTensorUtils.h>

#include <cusparse.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void topk_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> csrRowPtr,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> rowIdx,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> colIdx,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> value,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> outVal,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> outColIdx,
    int64_t k)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= rowIdx.size(0))
    {
        return;
    }

    const int row = rowIdx[idx];
    const int rank = idx - csrRowPtr[row];

    if (rank < k)
    {
        outVal[row][rank] = value[idx];
        outColIdx[row][rank] = colIdx[idx];
    }
}

template <typename scalar_t>
__global__ void dimmedian_idx_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> csrRowPtr,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> rowIdx,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> colIdx,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> value,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> X_argsort,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> X_rev_argsort,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> outIdx,
    const int64_t m,
    const int64_t d)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = idx / d;
    const int dim = idx % d;

    if ((row >= m) || (dim >= d))
    {
        return;
    }

    float weightSum = 0;
    for (int i = csrRowPtr[row]; i < csrRowPtr[row + 1]; i++)
    {
        weightSum += value[i];
    }

    int lastOrderIdx = -1;
    float cumSum = 0;
    int selectedElement = -3;
    for (int i = csrRowPtr[row]; cumSum < weightSum / 2; i++)
    {
        int currOrderIdx = std::numeric_limits<int>::max();
        for (int j = csrRowPtr[row]; j < csrRowPtr[row + 1]; j++)
        {
            int tempOrderIdx = X_rev_argsort[colIdx[j]][dim];
            if ((tempOrderIdx < currOrderIdx) && (tempOrderIdx > lastOrderIdx))
            {
                currOrderIdx = tempOrderIdx;
                selectedElement = j;
            }
        }
        lastOrderIdx = currOrderIdx;
        cumSum += value[selectedElement];
    }
    outIdx[row][dim] = colIdx[selectedElement];
}

void coo2csr(cusparseHandle_t handle, const int *cooRowIdx, int64_t nnz, int64_t m, int *csrRowPtr)
{
    TORCH_CHECK((m <= INT_MAX) && (nnz <= INT_MAX),
                "cusparseXcoo2csr only supports m, nnz with the bound [val] <= ",
                INT_MAX);

    TORCH_CUDASPARSE_CHECK(cusparseXcoo2csr(handle, cooRowIdx, (int)nnz, (int)m, csrRowPtr, CUSPARSE_INDEX_BASE_ZERO));
}

void csrsort(cusparseHandle_t handle,
             int64_t m,
             int64_t n,
             int64_t nnz,
             const int *csrRowPtr,
             int *valIdx,
             const float *val,
             float *outValSorted,
             const float *colIdx,
             float *outColIdxSorted)
{
    size_t pBufferSizeInBytes = 0;
    void *pBuffer = NULL;
    int *P = NULL;

    // step 1: allocate buffer
    cusparseXcsrsort_bufferSizeExt(handle, m, n, nnz, csrRowPtr, valIdx, &pBufferSizeInBytes);
    cudaMalloc(&pBuffer, sizeof(char) * pBufferSizeInBytes);

    // step 2: setup permutation vector P to identity
    cudaMalloc((void **)&P, sizeof(int) * nnz);
    cusparseCreateIdentityPermutation(handle, nnz, P);

    // step 3: sort CSR format
    cusparseMatDescr_t desc;
    cusparseCreateMatDescr(&desc);
    cusparseXcsrsort(handle, m, n, nnz, desc, csrRowPtr, valIdx, P, pBuffer);
    cusparseDestroyMatDescr(desc);

    // step 4: gather sorted csrVal
    cusparseSgthr(handle, nnz, val, outValSorted, P, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSgthr(handle, nnz, colIdx, outColIdxSorted, P, CUSPARSE_INDEX_BASE_ZERO);

    // step 5: free memory
    cudaFree(pBuffer);
    cudaFree(P);
}

void coosort(cusparseHandle_t handle,
             int64_t m,
             int64_t n,
             int64_t nnz,
             int *rowIdx,
             int *colIdx,
             int *P)
{
    size_t pBufferSizeInBytes = 0;
    void *pBuffer = NULL;
    // int *P = NULL;

    // step 1: allocate buffer
    cusparseXcoosort_bufferSizeExt(handle, m, n, nnz, rowIdx, colIdx, &pBufferSizeInBytes);
    cudaMalloc(&pBuffer, sizeof(char) * pBufferSizeInBytes);

    // step 2: setup permutation vector P to identity
    cusparseCreateIdentityPermutation(handle, nnz, P);

    // step 3: sort COO format
    cusparseXcoosortByRow(handle, m, n, nnz, rowIdx, colIdx, P, pBuffer);

    // step 4: free memory
    cudaFree(pBuffer);
}

std::vector<torch::Tensor> topk_forward_cuda(
    torch::Tensor edge_idx,
    torch::Tensor edge_weights,
    const int64_t n_edges,
    const int64_t k,
    const int n_threads = 1024)
{
    //sparse = sparse.coalesce();
    int64_t nnz = edge_weights.numel();
    int64_t m = n_edges;

    torch::Tensor values = edge_weights.to(torch::kFloat32);
    torch::Tensor rowIndices = edge_idx.select(0, 0);
    torch::Tensor colIndices = edge_idx.select(0, 1);

    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);

    // Convert COO to CSR
    torch::Tensor csrPtr = torch::empty({m + 1}, rowIndices.options().dtype(torch::kInt32));
    torch::Tensor rowIndicesInt = torch::empty({rowIndices.size(0)}, rowIndices.options().dtype(torch::kInt32));
    rowIndicesInt.copy_(rowIndices);
    torch::Tensor colIndicesInt = torch::empty({colIndices.size(0)}, rowIndices.options().dtype(torch::kInt32));
    colIndicesInt.copy_(colIndices);
    coo2csr(handle, rowIndicesInt.data_ptr<int32_t>(), nnz, m, csrPtr.data_ptr<int32_t>());

    // Convert values into idx preserving their order
    auto unique = torch::unique_dim(values.neg(), 0, true, true);
    int64_t u = std::get<0>(unique).size(0);
    torch::Tensor valueIdx = std::get<1>(unique).to(torch::kInt32);

    // Sort values per row
    torch::Tensor sortedValues = torch::empty({values.size(0)}, rowIndices.options().dtype(torch::kFloat32));
    // datatype hack...
    torch::Tensor sortedColIndicesInt = torch::empty({colIndicesInt.size(0)}, colIndicesInt.options().dtype(torch::kFloat32));
    csrsort(handle,
            m,
            u,
            nnz,
            csrPtr.data_ptr<int32_t>(),
            valueIdx.data_ptr<int32_t>(),
            values.data_ptr<float>(),
            sortedValues.data_ptr<float>(),
            colIndicesInt.to(torch::kFloat32).data_ptr<float>(), // datatype hack...
            sortedColIndicesInt.data_ptr<float>());
    cusparseDestroy(handle);
    // datatype hack...
    sortedColIndicesInt = sortedColIndicesInt.to(torch::kInt32);

    // Filter top k values
    const dim3 n_blocks(ceil((float)nnz / n_threads));
    torch::Tensor outVal = torch::zeros({m, k}, values.options());
    torch::Tensor outColIdx = torch::ones({m, k}, sortedColIndicesInt.options()).neg();
    AT_DISPATCH_INTEGRAL_TYPES(csrPtr.scalar_type(), "topk_forward_cuda", ([&] {
                                   topk_cuda_forward_kernel<scalar_t><<<n_blocks, n_threads>>>(
                                       csrPtr.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                       rowIndicesInt.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                       sortedColIndicesInt.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                       sortedValues.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                       outVal.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                       outColIdx.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       k);
                               }));
    return {outVal, outColIdx};
}

at::Tensor dimmedian_idx_forward_cuda(
    torch::Tensor X,
    torch::Tensor edge_idx,
    torch::Tensor edge_weights,
    const int nnz,
    const int n_rows,
    const int n_threads = 1024)
{
    torch::Tensor X_argsort = X.argsort(0).to(torch::kInt32);
    torch::Tensor X_rev_argsort = X_argsort.argsort(0).to(torch::kInt32);

    int64_t d = X.size(1);
    torch::Tensor values = edge_weights.to(torch::kFloat32);
    torch::Tensor rowIndices = edge_idx.select(0, 0);
    torch::Tensor colIndices = edge_idx.select(0, 1);

    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);

    // Convert COO to CSR
    torch::Tensor csrPtr = torch::empty({n_rows + 1}, rowIndices.options().dtype(torch::kInt32));
    torch::Tensor rowIndicesInt = torch::empty({rowIndices.size(0)}, rowIndices.options().dtype(torch::kInt32));
    rowIndicesInt.copy_(rowIndices);
    torch::Tensor colIndicesInt = torch::empty({colIndices.size(0)}, rowIndices.options().dtype(torch::kInt32));
    colIndicesInt.copy_(colIndices);
    coo2csr(handle, rowIndicesInt.data_ptr<int32_t>(), nnz, n_rows, csrPtr.data_ptr<int32_t>());

    const dim3 n_blocks(ceil((float)n_rows * d / n_threads));
    torch::Tensor outIdx = torch::full({n_rows, d}, -1, X.options().dtype(torch::kInt32));
    AT_DISPATCH_INTEGRAL_TYPES(csrPtr.scalar_type(), "dimmedian_idx_cuda_forward", ([&] {
                                   dimmedian_idx_cuda_forward_kernel<scalar_t><<<n_blocks, n_threads>>>(
                                       csrPtr.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                       rowIndicesInt.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                       colIndicesInt.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                       values.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                       X_argsort.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       X_rev_argsort.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       outIdx.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       n_rows,
                                       d);
                               }));
    return outIdx.to(torch::kInt64);
}
