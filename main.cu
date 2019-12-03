#include <stdio.h>
#define HIMATH_IMPL
#include "himath.h"

#define FUNC_PREFIX __device__

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

  __device__ FVec3 point_at_parameter(float t, const ray& ray)
  {
    FVec3 p = ray.origin + ray.dir*t;
    return p;
  }
}

namespace h_shape{
  struct sphere
  {
    FVec3 center;
    float radius;
  };
}

namespace h_intersect{
  struct ray_intersect_result
  {
    float t;
    bool hit;
    bool inside;
  };

  __device__ bool intersect_ray_sphere(const h_ray::ray& ray, const h_shape::sphere& sphere)
  {
    FVec3 oc = ray.origin - sphere.center;
    float a = fvec3_dot(ray.dir, ray.dir);
    float b = 2 * fvec3_dot(ray.dir, oc);
    float c = fvec3_dot(oc, oc) - sphere.radius*sphere.radius;

    // use quadratic formula to test p = (origin + dir*t) and sphere for all t
    float discriminant = b*b - 4*a*c;
    return (discriminant > 0);
  }
}

namespace h_app{
  __device__ FVec3 to_color(const h_ray::ray& ray)
  {
    FVec3 unit_dir = fvec3_normalize(ray.dir);
    float t = 0.5f * (unit_dir.y + 1.0f); // map y [-1, 1] to [0, 1]
    
    // h_shape::sphere sphere;
    // sphere.center = {0,0,-1};
    // sphere.radius = 0.5f;
    
    // if(h_intersect::intersect_ray_sphere(ray, sphere))
      // return {1,0,0};
            
    FVec3 white = {1.0f, 1.0f, 1.0f};
    FVec3 base = {0.5f, 0.7f, 1.0f};

    return  white*(1.0f - t) + base*t;
  }
}

__global__
void render(unsigned char* fb, int max_x, int max_y,
FVec3 lower_left_corner, FVec3 horizontal, FVec3 vertical, FVec3 origin)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if(i >= max_x || j >= max_y)
        return;
    
    int index = (j * max_x * 3) + (i*3);
    float u = float(i) / max_x;
    float v = float(j) / max_y;
    
    // dir's transition = from top-left to bottom-right
    h_ray::ray ray;
    ray.origin = origin;
    ray.dir = lower_left_corner + horizontal * u + vertical * v;
    FVec3 color = h_app::to_color(ray);
    
    // int samples = 100;
    // for(int s = 0; s < samples; ++s)
    // {
        // color.x += 1.0f / (samples+1);
        // color.y += 1.0f / (samples+1);
    // }
    // color.z = 0.2f;
    
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
    if( argc > 1 && argc <= 3)
    {
        nx = atoi(argv[1]);
        ny = atoi(argv[2]);
    }

    int num_pixels = nx * ny;

    unsigned char* fb;
    size_t fb_size = 3 * num_pixels * sizeof(unsigned char); 
    checkCudaError( cudaMallocManaged(&fb, fb_size) );

    FVec3 lower_left_corner = {-2, -1, -1};
    FVec3 horizontal = {4, 0, 0};
    FVec3 vertical = {0, 2, 0};
    FVec3 camera_pos = {0,0,0};

    int thread_x = 8;
    int thread_y = 8;
    int block_x = (nx + thread_x - 1)/thread_x;
    int block_y = (ny + thread_y - 1)/thread_y;
    dim3 blocks(block_x, block_y);
    dim3 threads(thread_x, thread_y);
    render<<<blocks, threads>>>( fb, nx, ny, 
    lower_left_corner, horizontal, vertical, camera_pos);

    checkCudaError( cudaGetLastError() );
    checkCudaError( cudaDeviceSynchronize() );
    print_ppm(fb, nx, ny);

    cudaFree(fb);

  return 0;  
}
