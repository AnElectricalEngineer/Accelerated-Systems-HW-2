#include "ex2.h"

#define NUM_OF_STREAMS 64
#define NUM_OF_THREADS 1024
#define N_IMAGES 10000

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
    cudaStream_t* streams;
    int* img_ids;

public:
    streams_server()
    {
        // TODO initialize context (memory buffers, streams, etc...)
    	CUDA_CHECK( cudaHostAlloc(&streams, NUM_OF_STREAMS*sizeof(cudaStream_t),0));
    	CUDA_CHECK( cudaHostAlloc(&img_ids, NUM_OF_STREAMS*sizeof(int),0));
        CUDA_CHECK( cudaMalloc(&dimg_in,N_IMAGES*IMG_WIDTH * IMG_HEIGHT) );
        CUDA_CHECK( cudaMalloc(&dimg_out, N_IMAGES*IMG_WIDTH * IMG_HEIGHT) );
        last_img_id = -1;
        for (int i = 0; i < NUM_OF_STREAMS; i++){
        	 CUDA_CHECK(cudaStreamCreate(&streams[i]));
             img_ids[i] = -1;
        }
    }

    ~streams_server() override
    {
        // TODO free resources allocated in constructor
        CUDA_CHECK( cudaFree(dimg_in) );
        CUDA_CHECK( cudaFree(dimg_out) );
        for (int i = 0; i < NUM_OF_STREAMS; i++)
        	 CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK(cudaFreeHost(streams));
        CUDA_CHECK(cudaFreeHost(img_ids));
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO place memory transfers and kernel invocation in streams if possible.

      //  if (last_img_id != -1)
       //     return false;
        int i =0;
        bool available_stream = false;
        for (;i<NUM_OF_STREAMS;i++){
        	if (img_ids[i]==-1){
        		available_stream = true;
        		break;
        	}
        }
        if (!available_stream) return false;

        CUDA_CHECK( cudaMemcpyAsync(dimg_in + img_id*IMG_WIDTH * IMG_HEIGHT, img_in, IMG_WIDTH * IMG_HEIGHT, cudaMemcpyHostToDevice, streams[i]));
        process_image_kernel<<<1, 1024, 0, streams[i]>>>(dimg_in + img_id*IMG_WIDTH * IMG_HEIGHT, dimg_out + img_id*IMG_WIDTH * IMG_HEIGHT);
        CUDA_CHECK( cudaMemcpyAsync(img_out, dimg_out + img_id*IMG_WIDTH * IMG_HEIGHT, IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToHost, streams[i]));
      //  last_img_id = img_id;
        img_ids[i] = img_id;
        return true;
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) streams for any completed requests.

  //      if (last_img_id < 0)
    //        return false;
        for (int strm = 0; strm<NUM_OF_STREAMS; strm++){
        	if (img_ids[strm]==-1) continue;
			cudaError_t status = cudaStreamQuery(streams[strm]);
			switch (status) {
			case cudaSuccess:
				*img_id = img_ids[strm];
				img_ids[strm] = -1;
		//		last_img_id = -1;
				return true;
			case cudaErrorNotReady:
				continue;
			default:
				CUDA_CHECK(status);
				continue;
			}
        }
        return false;
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
