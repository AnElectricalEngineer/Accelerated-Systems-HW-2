#include "ex2.h"
#include <cuda/atomic>

#define NUM_OF_STREAMS 64
#define NUM_OF_THREADS 1024
#define N_IMAGES 10000
//typedef cuda::atomic<size_t> cpu_gpu_atomic;

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
__device__ void process_image_kernel(uchar *in, uchar *out) {
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

__global__ void process_image_kernel_streams(uchar *in, uchar *out) {
	process_image_kernel(in, out);
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
        process_image_kernel_streams<<<1, 1024, 0, streams[i]>>>(dimg_in + img_id*IMG_WIDTH * IMG_HEIGHT, dimg_out + img_id*IMG_WIDTH * IMG_HEIGHT);
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

//////////////////////////***************************///////////////////////////////
#define SHARED_MEM_PER_THREAD_BLOCK 1304
#define REG_PER_THREAD 32

class gpu_task{
public:
	int    _img_id;
	uchar* _img_in;
	uchar* _img_out;

	__host__ __device__ gpu_task(int img_id, uchar *img_in, uchar *img_out) :
	                	_img_id(img_id), _img_in(img_in), _img_out(img_out){}

	__host__ __device__ gpu_task() :
	                	_img_id(-1), _img_in(nullptr), _img_out(nullptr){}
};




class queue{
	static const size_t TASKS_PER_QUEUE = 16;
	gpu_task _tasks[TASKS_PER_QUEUE];
	cuda::atomic<size_t, cuda::thread_scope_system>  _head;
	cuda::atomic<size_t, cuda::thread_scope_system>  _tail;

public:
	queue(): _head(0), _tail(0){}

	__host__ __device__ void push(gpu_task task){
		size_t tail = _tail.load(cuda::memory_order_relaxed);
		//printf("#########17###############");
		while ((tail - _head.load(cuda::memory_order_acquire)) == TASKS_PER_QUEUE);
		//printf("#########18###############");
		_tasks[_tail % TASKS_PER_QUEUE] = task;
		_tail.store(tail+1, cuda::memory_order_release);
	}

	__host__ __device__ gpu_task pop(){
		size_t head = _head.load(cuda::memory_order_relaxed);
		while (head == _tail.load(cuda::memory_order_acquire));
		gpu_task task = _tasks[ head % TASKS_PER_QUEUE];
		_head.store(head+1, cuda::memory_order_release);
		return task;
	}

	__host__ __device__ bool is_queue_empty(){
		size_t head = _head.load(cuda::memory_order_relaxed);
		if (head == _tail.load(cuda::memory_order_acquire))
				return true;
		return false;
	}

	__host__ __device__ bool is_queue_full(){
		size_t tail = _tail.load(cuda::memory_order_relaxed);
		if ((tail - _head.load(cuda::memory_order_acquire)) == TASKS_PER_QUEUE)
				return true;
		return false;
	}

};




__global__ void process_image_queue_kernel(	queue** cpu_to_gpu, queue** gpu_to_cpu, cuda::atomic<bool>* running){
	__shared__ uchar* img_in;
	__shared__ uchar* img_out;
	__shared__ int img_id;
	__shared__ bool get_out;

	int thread_id = threadIdx.x;
	int block_id  =  blockIdx.x;
	if (!thread_id){
		get_out= false;
	}
	__syncthreads();

	while (true) {
		if (!thread_id){
			bool run = running->load(cuda::memory_order_acquire);
			bool empty = cpu_to_gpu[block_id]->is_queue_empty();
		    printf("#########runnnn: %d###############\n",run);
			printf("#########empty: %d###############\n",empty);
			if ( empty && !(run)) {
				printf("#########5###############");
				get_out=true;
				running->store(true,cuda::memory_order_release);
				//printf("#########6###############");
			}
			else{
				//printf("#########7###############");
				gpu_task task = cpu_to_gpu[block_id]->pop();
				//printf("#########8###############");
				img_in = task._img_in;
				img_out = task._img_out;
				img_id = task._img_id;
			}
		}
		 __syncthreads();
		if (get_out){
			printf("#########9###############");
			return;
		}

        process_image_kernel(img_in, img_out);
        //printf("#########10###############");
        __syncthreads();

		if (!thread_id){
			//printf("#########11###############");
			gpu_to_cpu[block_id]->push(gpu_task(img_id, img_in, img_out));
			//printf("#########12###############");

		}
		 __syncthreads();
	}
}


class queue_server : public image_processing_server
{
private:
	queue** cpu_to_gpu;
	queue** gpu_to_cpu;
	int num_of_blocks;
	cuda::atomic<bool>* running;


public:
    queue_server(int threads)
    {
    	cudaDeviceProp gpu_prop;
    	cudaGetDeviceProperties(&gpu_prop, 0);
    	int max_threads_per_core = gpu_prop.maxThreadsPerMultiProcessor;
    	int num_of_cores =  gpu_prop.multiProcessorCount;
    	num_of_blocks = (num_of_cores * max_threads_per_core) /  threads;
    	int shared_mem_per_core = gpu_prop.sharedMemPerMultiprocessor;
    	int regs_per_core = gpu_prop.regsPerMultiprocessor;
    	//size_t max_blocks_per_core = gpu_prop.maxBlocksPerMultiProcessor;

    /*	if(num_of_blocks >  num_of_cores * max_blocks_per_core)
    		num_of_blocks = num_of_cores * max_blocks_per_core;*/

    	if (num_of_blocks > num_of_cores * (int)(shared_mem_per_core / SHARED_MEM_PER_THREAD_BLOCK))
    		num_of_blocks = num_of_cores * (int)(shared_mem_per_core / SHARED_MEM_PER_THREAD_BLOCK);

    	if (num_of_blocks > num_of_cores * (int)(regs_per_core / (threads * REG_PER_THREAD)))
    		num_of_blocks = num_of_cores * (int)(regs_per_core / (threads * REG_PER_THREAD));

    	CUDA_CHECK(cudaMallocHost(&cpu_to_gpu, num_of_blocks * sizeof(queue*)));
    	CUDA_CHECK(cudaMallocHost(&gpu_to_cpu, num_of_blocks * sizeof(queue*)));
    //	::new(cpu_to_gpu) queue*();
   // 	::new(gpu_to_cpu) queue*();

    	for (int i =0; i< num_of_blocks;i++){
        	::new(cpu_to_gpu+i) queue*();
        	::new(gpu_to_cpu+i) queue*();
        	CUDA_CHECK(cudaMallocHost((cpu_to_gpu + i), sizeof(queue)));
        	CUDA_CHECK(cudaMallocHost((gpu_to_cpu + i), sizeof(queue)));
        	::new(*(cpu_to_gpu + i)) queue();
        	::new(*(gpu_to_cpu + i)) queue();
    	}
    	CUDA_CHECK(cudaMallocHost(&running, sizeof(cuda::atomic<bool>)));
    	::new(running) cuda::atomic<bool>(true);
    	printf("#########num_of_blocks: %d###############",num_of_blocks);
    	//process_image_queue_kernel<<<num_of_blocks, threads>>>(cpu_to_gpu, gpu_to_cpu, running);
    	process_image_queue_kernel<<<1, 1>>>(cpu_to_gpu, gpu_to_cpu, running);
    	//printf("#########15###############");
    }

    ~queue_server() override
    {
    	printf("#########1###############");

    	running->exchange(false, cuda::memory_order_acq_rel);
    	printf("#########4###############");
    	CUDA_CHECK(cudaDeviceSynchronize());
    	printf("#########2###############");

    	for (int i =0; i< num_of_blocks;i++){
        	cpu_to_gpu[i]-> ~queue();
        	gpu_to_cpu[i]-> ~queue();
        	CUDA_CHECK(cudaFreeHost(cpu_to_gpu[i]));
        	CUDA_CHECK(cudaFreeHost(gpu_to_cpu[i]));
    	}

        //cpu_to_gpu-> ~queue*();
    	//gpu_to_cpu-> ~queue*();
    	running->~atomic<bool>();
    	printf("#########a###############");
    	CUDA_CHECK(cudaFreeHost(cpu_to_gpu));
    	CUDA_CHECK(cudaFreeHost(gpu_to_cpu));
    	CUDA_CHECK(cudaFreeHost(running));
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
    	for (int i =0; i< num_of_blocks;i++){
    		if (cpu_to_gpu[i]->is_queue_full())
    			continue;

    		cpu_to_gpu[i]->push(gpu_task(img_id, img_in, img_out));
    		return true;
    	}
        return false;
    }

    bool dequeue(int *img_id) override
    {
    	for (int i =0; i< num_of_blocks;i++){
    		if (gpu_to_cpu[i]->is_queue_empty())
    			continue;

    		gpu_task task = gpu_to_cpu[i]->pop();
    		*img_id = task._img_id;
    		//printf("#########3###############");
    		return true;
    	}
        return false;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
