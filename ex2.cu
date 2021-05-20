#include "ex2.h"

__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int increment;

    for (int stride = 1; stride < min(blockDim.x, arr_size); stride *= 2) {
        if (tid >= stride && tid < arr_size) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < arr_size) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
}

// Example single-threadblock kernel for processing a single image.
// Feel free to change it.
__global__ void process_image_kernel(uchar *in, uchar *out) {
    __shared__ int histogram[256];
    __shared__ uchar map[256];

    int tid = threadIdx.x;

    if (tid < 256) {
        histogram[tid] = 0;
    }
    __syncthreads();

    for (int i = tid; i < IMG_HEIGHT * IMG_HEIGHT; i += blockDim.x)
        atomicAdd(&histogram[in[i]], 1);

    __syncthreads();

    prefix_sum(histogram, 256);

    if (tid < 256) {
        float map_value = float(histogram[tid]) / (IMG_WIDTH * IMG_HEIGHT);
        map[tid] = ((uchar)(N_COLORS * map_value)) * (256 / N_COLORS);
    }

    __syncthreads();

    for (int i = tid; i < IMG_WIDTH * IMG_HEIGHT; i += blockDim.x) {
        out[i] = map[in[i]];
    }
}

class streams_server : public image_processing_server
{
private:
    // TODO define stream server context (memory buffers, streams, etc...)

    // Feel free to change the existing memory buffer definitions.
    uchar *dimg_in;
    uchar *dimg_out;
    int last_img_id;

public:
    streams_server()
    {
        // TODO initialize context (memory buffers, streams, etc...)
        CUDA_CHECK( cudaMalloc(&dimg_in, IMG_WIDTH * IMG_HEIGHT) );
        CUDA_CHECK( cudaMalloc(&dimg_out, IMG_WIDTH * IMG_HEIGHT) );
        last_img_id = -1;
    }

    ~streams_server() override
    {
        // TODO free resources allocated in constructor
        CUDA_CHECK( cudaFree(dimg_in) );
        CUDA_CHECK( cudaFree(dimg_out) );
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO place memory transfers and kernel invocation in streams if possible.

        if (last_img_id != -1)
            return false;

        CUDA_CHECK( cudaMemcpyAsync(dimg_in, img_in, IMG_WIDTH * IMG_HEIGHT, cudaMemcpyHostToDevice ));
        process_image_kernel<<<1, 1024>>>(dimg_in, dimg_out);
        CUDA_CHECK( cudaMemcpyAsync(img_out, dimg_out, IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToHost ));
        last_img_id = img_id;
        return true;
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) streams for any completed requests.

        if (last_img_id < 0)
            return false;

        cudaError_t status = cudaStreamQuery(0);
        switch (status) {
        case cudaSuccess:
            *img_id = last_img_id; // TODO return the img_id of the request that was completed.
            last_img_id = -1;
            return true;
        case cudaErrorNotReady:
            return false;
        default:
            CUDA_CHECK(status);
            return false;
        }
    }
};

std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}

class queue_server : public image_processing_server
{
private:
    // TODO define queue server context (memory buffers, etc...)
public:
    queue_server(int threads)
    {
        // TODO initialize host state
        // TODO launch GPU producer-consumer kernel with given number of threads
    }

    ~queue_server() override
    {
        // TODO free resources allocated in constructor
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO push new task into queue if possible
        return true;
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) the producer-consumer queue for any responses.
        *img_id = 0; // TODO return the img_id of the request that was completed.
        return true;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
