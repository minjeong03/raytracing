#include <stdio.h>
#define HIMATH_IMPL
#include "himath.h"
#include <curand_kernel.h>


#define checkCudaError(val) check_cuda( (val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, 
char const *const func, const char* file, int const line)
{
    if( result != cudaSuccess ) {
        fprintf(stderr, "CUDA error = '%i' %s at %s : %i '%s'\n", unsigned(result),
        cudaGetErrorString(result),
        file, line, func);
        
        cudaDeviceReset();
        exit(99);
    }
}


__device__ FVec3 random_in_unit_sphere(curandState* local_rand_state)
{
    FVec3 rand = {0,0,0};
    do
    {        
        rand.x = 2.f * curand_uniform(local_rand_state) - 1.f;
        rand.y = 2.f * curand_uniform(local_rand_state) - 1.f;
        rand.z = 2.f * curand_uniform(local_rand_state) - 1.f;   
    }while(fvec3_dot(rand, rand) >= 1);
    
    return rand;
}


namespace h_ray{
  struct ray
  {
    FVec3 origin;
    FVec3 dir;
  };
  
  __device__ ray create_ray(FVec3 origin, FVec3 dir)
  {
      ray r;
      r.origin = origin;
      r.dir = fvec3_normalize(dir);
      return r;
  }

  __device__ FVec3 point_at_parameter(const ray& ray, float t)
  {
    return ray.origin + ray.dir*t;
  }
}

namespace h_shape{
  struct sphere
  {
    FVec3 center;
    float radius;
  };
}

namespace h_hit{
  struct hit_record
  {
      float t;
      FVec3 p;
      FVec3 n;
  };
  
  __device__ bool 
  hit_sphere(const h_ray::ray& ray, const h_shape::sphere& sphere, 
  float t_min, float t_max, hit_record& record)
  {
    FVec3 oc = ray.origin - sphere.center;
    float a = fvec3_dot(ray.dir, ray.dir);
    float b = 2 * fvec3_dot(ray.dir, oc);
    float c = fvec3_dot(oc, oc) - sphere.radius*sphere.radius;

    // use quadratic formula to test p = (origin + dir*t) and sphere for all t
    float discriminant = b*b - 4*a*c;
    if(discriminant < 0)
        return false;
    
    float sqrt_d = sqrtf(discriminant);
    float t1 = (-b-sqrt_d) / 2*a;
    float t2 = (-b+sqrt_d) / 2*a;
    if(t1 > t_max || t2 < t_min)
        return false;
    
    record.t = (t1 > t_min? t1:t2);
    
    // These variables below can be computed by t, ray and shape.
    record.p = h_ray::point_at_parameter(ray, record.t);
    record.n = (record.p - sphere.center) / sphere.radius;
    return true;
  }
  
  
  // Check if a ray intersects any objects in the world,
  // returns intersection point, normal, its material
  __device__ bool
  hit_world(const h_ray::ray& ray, h_shape::sphere* world, int obj_size,
  float t_min, float t_max, hit_record& record) 
  {
    bool hit = false;
    float closest_t = t_max;
    hit_record temp;
    for(int i = 0; i < obj_size; ++i)
    {
        if(hit_sphere(ray, world[i], t_min, t_max, temp))
        {
            hit = true;
            if(temp.t < closest_t)
            {
                record = temp;
                closest_t = record.t;
            }
        }
    }
    return hit;
  }
}


// Path tracing
__device__ 
FVec3 to_color(const h_ray::ray& init_ray, h_shape::sphere* world, int obj_size,
                 int max_depth, curandState* local_rand_state)
{
    FVec3 white = {1.0f, 1.0f, 1.0f};              
    FVec3 ambient = {0.5f, 0.7f, 1.0f};

    h_hit::hit_record record;
    h_ray::ray ray = init_ray;
    
    float coef_attenuation = 0.5f;
    float attenuation = 1.f;
    
    for(int depth = 0; depth < max_depth; ++depth)
    {
        // t_min = 0.0000001 --> artifact 
        // t_min = 0.001
        if( !h_hit::hit_world(ray, world, obj_size, 0.001f, 100000.f, record) ) 
        {            
            float t = 0.5f * (ray.dir.y + 1.0f);
            FVec3 color = white*(1.0f - t) + ambient*t;
            return color * attenuation;
        }
        
        FVec3 diffused = random_in_unit_sphere(local_rand_state);
        ray = h_ray::create_ray(record.p, record.n + diffused);
    
        attenuation *= coef_attenuation;
    }
       
    FVec3 black = {0, 0, 0};
    return black;
}

// Render a pixel by casting a ray from camera to the point on viewport 
// computed by pixel position and framebuffer size
__global__
void render(unsigned char* fb, int max_x, int max_y, int ns,
FVec3 lower_left_corner, FVec3 horizontal, FVec3 vertical, FVec3 origin,
h_shape::sphere** world, int* size, int max_depth,
curandState* rand_state)
{
    float i = threadIdx.x + blockIdx.x * blockDim.x;
    float j = threadIdx.y + blockIdx.y * blockDim.y;

    if(i >= max_x || j >= max_y)
        return;
    
    int index = (j * max_x) + i;
    curandState* local_rand_state = rand_state+index;
    
    FVec3 color = {0,0,0};
    float max_xf = max_x;
    float max_yf = max_y;
    
    for(int s = 0; s < ns; ++s)
    {
        float u = ( i + 1.f - curand_uniform(local_rand_state) ) / max_xf;
        float v = ( j + 1.f - curand_uniform(local_rand_state) ) / max_yf;

        h_ray::ray ray = h_ray::create_ray(
            origin, lower_left_corner + horizontal * u + vertical * v);
        
        color += to_color(ray, *world, *size, max_depth, local_rand_state);
    }
    color /= float(ns);
    color.x = sqrtf(color.x);
    color.y = sqrtf(color.y);
    color.z = sqrtf(color.z);
    
    fb[index*3+0] = unsigned char(255.99f * color.x);
    fb[index*3+1] = unsigned char(255.99f * color.y);
    fb[index*3+2] = unsigned char(255.99f * color.z);
}


__global__
void create_world(h_shape::sphere** world, int* obj_size)
{
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        *obj_size = 2;
        *world = (h_shape::sphere*)malloc(sizeof(h_shape::sphere) * (*obj_size));
        
        h_shape::sphere* s = *world;
        s[0].center = {0, 0, -1};
        s[0].radius = 0.5f;
        s[1].center = {0, -100.5f, -1};
        s[1].radius = 100.f;
    }
}

__global__
void free_world(h_shape::sphere** world)
{
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        free(*world);
    }
}

__global__
void rand_init(curandState* rand_state, int max_x, int max_y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if(i >= max_x || j >= max_y)
        return;

    int index = (j * max_x) + i;
    
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, index, 0, &rand_state[index]);
}

void print_ppm(unsigned char* fb, int w, int h)
{
    printf("P3\n%i %i\n255\n", w, h);
    for (int j = h-1; j >= 0; j--) {
        for(int i = 0; i < w; i++)   {
          int index = (j*w*3) + (i*3);
          printf("%i %i %i\n", fb[index+0], fb[index+1], fb[index+2]);
        }
    }
}

int main(int argc, char **argv) {
    
    int nx = 1920;
    int ny = 1080;
    int ns = 100;
    if( argc >= 3)
    {
        nx = atoi(argv[1]);
        ny = atoi(argv[2]);
    }
    if( argc == 4)
    {
        ns = atoi(argv[3]);
    }

    // Allocate framebuffer in the unified memory
    int num_pixels = nx * ny;
    unsigned char* fb;
    size_t fb_size = 3 * num_pixels * sizeof(unsigned char); 
    checkCudaError( cudaMallocManaged(&fb, fb_size) );
    
    
    // Allocate random number state(?) on the device global memory
    curandState *dev_rand_state;
    checkCudaError( cudaMalloc(&dev_rand_state, num_pixels*sizeof(curandState)) );
    
    
    // Allocate variables that are passed to and used in the device
    h_shape::sphere** dev_world;
    int* dev_num_obj;
    checkCudaError( cudaMalloc(&dev_world, sizeof(h_shape::sphere**)) );
    checkCudaError( cudaMalloc(&dev_num_obj, sizeof(int)) );
  
    // Initialize objects in the world
    create_world<<<1,1>>>(dev_world, dev_num_obj);
    checkCudaError( cudaGetLastError() );
    checkCudaError( cudaDeviceSynchronize() );
    
    
    // Initialize the camera and the view variables in the world
    float aspect_ratio = float(nx)/ ny;
    float h = 2;
    float w = aspect_ratio * h; 
    FVec3 lower_left_corner = {-w/2.f, -h/2.f, -1};
    FVec3 horizontal = {w, 0, 0};
    FVec3 vertical = {0, h, 0};
    FVec3 camera_pos = {0,0,0};

    
    // Determine the number of threads and blocks
    int thread_x = 8;
    int thread_y = 8;
    int block_x = (nx + thread_x - 1)/thread_x;
    int block_y = (ny + thread_y - 1)/thread_y;
    
    fprintf(stderr, "blocks = (%i, %i)", block_x, block_y);
    
    dim3 blocks(block_x, block_y);
    dim3 threads(thread_x, thread_y);
    
    // Initialize random state
    rand_init<<<blocks, threads>>>(dev_rand_state, nx, ny);
    checkCudaError( cudaGetLastError() );
    checkCudaError( cudaDeviceSynchronize() );
    
    // Render objs in the framebuffer
    int max_depth = 25;
    render<<<blocks, threads>>>( fb, nx, ny, ns,
                                 lower_left_corner, horizontal, vertical, camera_pos, 
                                 dev_world, dev_num_obj, max_depth,
                                 dev_rand_state);
    checkCudaError( cudaGetLastError() );
    checkCudaError( cudaDeviceSynchronize() );
  
  
    // Deallocate objs in world
    free_world<<<1,1>>>(dev_world);
    checkCudaError( cudaGetLastError() );
    checkCudaError( cudaDeviceSynchronize() );
      
      
    // Deallocate heap world variables
    checkCudaError( cudaFree(dev_world) );
    checkCudaError( cudaFree(dev_num_obj) );
    
    // Deallocate random states
    checkCudaError( cudaFree(dev_rand_state) );
    
    print_ppm(fb, nx, ny);
    
    // Deallocate framebuffer
    checkCudaError( cudaFree(fb) );

    return 0;  
}
