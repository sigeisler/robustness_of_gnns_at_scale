#include <torch/extension.h>

#include <ATen/cuda/CUDAUtils.h>
#include <ATen/SparseTensorUtils.h>

#include <cusparse.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

#include <vector>

template <typename scalar_t>
__global__ void topk_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> csrPtr,
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
    const int rank = idx - csrPtr[row];

    if (rank < k)
    {
        outVal[row][rank] = value[idx];
        outColIdx[row][rank] = colIdx[idx];
    }
}

__device__ void merge(float key[], int start, int mid, int end, float val[], int perm[])
{
    int start2 = mid + 1;
 
    // If the direct merge is already sorted
    if (key[mid] <= key[start2]) {
        return;
    }
 
    // Two pointers to maintain start
    // of both arrays to merge
    while (start <= mid && start2 <= end) {
 
        // If element 1 is in right place
        if (key[start] <= key[start2]) {
            start++;
        }
        else {
            float currentKey = key[start2];
            float currentValue = val[start2];
            int currentPerm = perm[start2];
            int index = start2;
 
            // Shift all the elements between element 1
            // element 2, right by 1.
            while (index != start) {
                key[index] = key[index - 1];
                val[index] = val[index - 1];
                perm[index] = perm[index - 1];
                index--;
            }
            key[start] = currentKey;
            val[start] = currentValue;
            perm[start] = currentPerm;
 
            // Update all the pointers
            start++;
            mid++;
            start2++;
        }
    }
}
 
/* l is for left index and r is right index of the
   sub-array of arr to be sorted */
__device__ void mergeSort(float key[], int start, int end, float val[], int perm[])
{
    if (start < end) {
 
        // Same as (l + r) / 2, but avoids overflow
        // for large l and r
        int mid = start + (end - start) / 2;
 
        // Sort first and second halves
        mergeSort(key, start, mid, val, perm);
        mergeSort(key, mid + 1, end, val, perm);
 
        merge(key, start, mid, end, val, perm);
    }
}

template <typename scalar_t>
__global__ void dimmedian_idx_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> csrPtr,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> rowIdx,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> colIdx,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> attr,
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

    const int start = csrPtr[row];
    const int length = csrPtr[row + 1] - start;

    float* localWeights = (float*) malloc(length * sizeof(float));
    float* localAttr = (float*) malloc(length * sizeof(float));
    int* localPerm = (int*) malloc(length * sizeof(int));

    if (!localWeights || !localAttr || !localPerm) {
        printf("Start: %d dim: %d length: %d localWeights: %p localAttr: %p localPerm: %p\n", start, dim, length, localWeights, localAttr, localPerm);
    }

    for (int i = 0; i < length; i++)
    {
        localWeights[i] = weight[i + start];
        localAttr[i] = attr[colIdx[i + start]][dim];
        localPerm[i] = i;
    }
    
    // Originally planned approach that lead to errors due to severe resource allocation
    //thrust::sort_by_key(thrust::seq, localAttr, localAttr + length, localWeights);
    //thrust::inclusive_scan(thrust::seq, localWeights, localWeights + length, localWeights);

    mergeSort(localAttr, 0, length - 1, localWeights, localPerm);
    free(localAttr);

    float weightSum = 0;
    for (int i = 0; i < length; i++)
    {
        localWeights[i] += weightSum;
        weightSum = localWeights[i];
    }

    int medianIndex = -1;
    for (int i = 0; localWeights[i] < weightSum / 2; i++)
    {
        medianIndex = i;
    }
    if (medianIndex < length - 1) {
        medianIndex += 1;
    }

    free(localWeights);

    outIdx[row][dim] = colIdx[start + localPerm[medianIndex]];

    free(localPerm);
}

void coo2csr(cusparseHandle_t handle, const int *cooRowIdx, int64_t nnz, int64_t m, int *csrPtr)
{
    TORCH_CHECK((m <= INT_MAX) && (nnz <= INT_MAX),
                "cusparseXcoo2csr only supports m, nnz with the bound [val] <= ",
                INT_MAX);

    TORCH_CUDASPARSE_CHECK(cusparseXcoo2csr(handle, cooRowIdx, (int)nnz, (int)m, csrPtr, CUSPARSE_INDEX_BASE_ZERO));
}

void csrsort(cusparseHandle_t handle,
             int64_t m,
             int64_t n,
             int64_t nnz,
             const int *csrPtr,
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
    cusparseXcsrsort_bufferSizeExt(handle, m, n, nnz, csrPtr, valIdx, &pBufferSizeInBytes);
    cudaMalloc(&pBuffer, sizeof(char) * pBufferSizeInBytes);

    // step 2: setup permutation vector P to identity
    cudaMalloc((void **)&P, sizeof(int) * nnz);
    cusparseCreateIdentityPermutation(handle, nnz, P);

    // step 3: sort CSR format
    cusparseMatDescr_t desc;
    cusparseCreateMatDescr(&desc);
    cusparseXcsrsort(handle, m, n, nnz, desc, csrPtr, valIdx, P, pBuffer);
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
    const int n_threads = 256)
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
    torch::Tensor attr,
    torch::Tensor edge_idx,
    torch::Tensor edge_weights,
    const int nnz,
    const int n_rows,
    const int n_threads = 1024)
{
    int64_t d = attr.size(1);
    torch::Tensor values = edge_weights.to(torch::kFloat32);
    torch::Tensor rowIndices = edge_idx.select(0, 0);
    torch::Tensor colIndices = edge_idx.select(0, 1);

    size_t limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
    printf("cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize): %d\n", limit);
    if(limit < 4*1024*1024*1024) {
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, 4*1024*1024*1024);
    }

    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);

    // Convert COO to CSR
    torch::Tensor csrPtr = torch::empty({n_rows + 1}, rowIndices.options().dtype(torch::kInt32));
    torch::Tensor rowIndicesInt = torch::empty({rowIndices.size(0)}, rowIndices.options().dtype(torch::kInt32));
    rowIndicesInt.copy_(rowIndices);
    torch::Tensor colIndicesInt = torch::empty({colIndices.size(0)}, rowIndices.options().dtype(torch::kInt32));
    colIndicesInt.copy_(colIndices);
    coo2csr(handle, rowIndicesInt.data_ptr<int32_t>(), nnz, n_rows, csrPtr.data_ptr<int32_t>());

    const dim3 n_blocks(ceil((float) n_rows * d / n_threads));
    torch::Tensor outIdx = torch::full({n_rows, d}, -1, attr.options().dtype(torch::kInt32));
    AT_DISPATCH_INTEGRAL_TYPES(csrPtr.scalar_type(), "dimmedian_idx_cuda_forward", ([&] {
                                   dimmedian_idx_cuda_forward_kernel<scalar_t><<<n_blocks, n_threads>>>(
                                       csrPtr.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                       rowIndicesInt.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                       colIndicesInt.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                       values.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                       attr.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                       outIdx.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       n_rows,
                                       d);
                               }));
    return outIdx.to(torch::kInt64);
}

