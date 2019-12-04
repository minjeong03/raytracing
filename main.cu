#include <stdio.h>
#define HIMATH_IMPL
#include "himath.h"

#define DEVICE_FUNC __device__

#define checkCudaError(val) check_cuda( (val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char* file, int const line)
{
    if(result) {
        fprintf(stderr, "CUDA error = %i at %s : %i '%s'\n", unsigned(result),
        file, line, func);
        
        cudaDeviceReset();
        exit(99);
    }
}

namespace h_ray{
  struct ray
  {
    FVec3 origin;
    FVec3 dir;
  };

  DEVICE_FUNC FVec3 point_at_parameter(const ray& ray, float t)
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
  DEVICE_FUNC bool 
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
    record.p = h_ray::point_at_parameter(ray, record.t);
    record.n = (record.p - sphere.center) / sphere.radius;
    return true;
  }
}

namespace h_app{
  DEVICE_FUNC FVec3 to_color(const h_ray::ray& ray, h_shape::sphere* world, int obj_size)
  {
    FVec3 white = {1.0f, 1.0f, 1.0f};
   
    h_hit::hit_record record;
    if(h_hit::hit_sphere(ray, world[0], 0.000001f, 100000.f, record)) {
        
        return (record.n + white) * 0.5f;
    }
    
    FVec3 unit_dir = fvec3_normalize(ray.dir);
    float t = 0.5f * (unit_dir.y + 1.0f);
      
    FVec3 base = {0.5f, 0.7f, 1.0f};

    return  white*(1.0f - t) + base*t;
  }
}

__global__
void create_world(h_shape::sphere** world, int* obj_size)
{
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        *obj_size = 1;
        *world = (h_shape::sphere*)malloc(sizeof(h_shape::sphere) * (*obj_size));
        
        h_shape::sphere* s = *world;
        s[0].center = {0, 0, -1};
        s[0].radius = 0.5f;
        //s[1].center = {0, -100.5f, -1};
        //s[1].radius = 100.f;
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
void render(unsigned char* fb, int max_x, int max_y,
FVec3 lower_left_corner, FVec3 horizontal, FVec3 vertical, FVec3 origin,
h_shape::sphere** world, int* size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if(i >= max_x || j >= max_y)
        return;
    
    int index = (j * max_x * 3) + (i*3);
    float u = float(i) / max_x;
    float v = float(j) / max_y;
    
    h_ray::ray ray;
    ray.origin = origin;
    ray.dir = fvec3_normalize(lower_left_corner + horizontal * u + vertical * v);
    
    FVec3 color = h_app::to_color(ray, *world, *size);
    
    
    fb[index+0] = unsigned char(255.99f * color.x);
    fb[index+1] = unsigned char(255.99f * color.y);
    fb[index+2] = unsigned char(255.99f * color.z);
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


// right handed coordinate
// +x-right
// +y-up
// +z-out of screen
int main(int argc, char **argv) {
    int nx = 1920;
    int ny = 1080;
    if( argc == 3)
    {
        nx = atoi(argv[1]);
        ny = atoi(argv[2]);
    }

    int num_pixels = nx * ny;
    unsigned char* fb;
    size_t fb_size = 3 * num_pixels * sizeof(unsigned char); 
    checkCudaError( cudaMallocManaged(&fb, fb_size) );
    
    h_shape::sphere** dev_world;
    int* dev_num_obj;
    checkCudaError( cudaMalloc(&dev_world, sizeof(h_shape::sphere**)) );
    checkCudaError( cudaMalloc(&dev_num_obj, sizeof(int)) );
  
    create_world<<<1,1>>>(dev_world, dev_num_obj);
    checkCudaError( cudaGetLastError() );
    checkCudaError( cudaDeviceSynchronize() );
    
    float aspect_ratio = float(nx)/ ny;
    float h = 2;
    float w = aspect_ratio * h; 
    FVec3 lower_left_corner = {-w/2.f, -h/2.f, -1};
    FVec3 horizontal = {w, 0, 0};
    FVec3 vertical = {0, h, 0};
    FVec3 camera_pos = {0,0,0};

    int thread_x = 8;
    int thread_y = 8;
    int block_x = (nx + thread_x - 1)/thread_x;
    int block_y = (ny + thread_y - 1)/thread_y;
    dim3 blocks(block_x, block_y);
    dim3 threads(thread_x, thread_y);
    render<<<blocks, threads>>>( fb, nx, ny, 
                                 lower_left_corner, horizontal, vertical, camera_pos, 
                                 dev_world, dev_num_obj);
    checkCudaError( cudaGetLastError() );
    checkCudaError( cudaDeviceSynchronize() );
  
    free_world<<<1,1>>>(dev_world);
    checkCudaError( cudaGetLastError() );
    checkCudaError( cudaDeviceSynchronize() );
      
    checkCudaError( cudaFree(dev_world) );
    checkCudaError( cudaFree(dev_num_obj) );
    
    
    print_ppm(fb, nx, ny);
    cudaFree(fb);

    return 0;  
}
