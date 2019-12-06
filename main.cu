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

__device__ bool 
ray_sphere(const h_ray::ray& ray, const FVec3& center, float radius, 
float t_min, float t_max, float& t)
{
    FVec3 oc = ray.origin - center;
    float a = fvec3_dot(ray.dir, ray.dir);
    float b = 2 * fvec3_dot(ray.dir, oc);
    float c = fvec3_dot(oc, oc) - radius*radius;

    // use quadratic formula to test p = (origin + dir*t) and sphere for all t
    float discriminant = b*b - 4*a*c;
    if(discriminant < 0)
        return false;

    float sqrt_d = sqrtf(discriminant);
    float t1 = (-b-sqrt_d) / 2*a;
    float t2 = (-b+sqrt_d) / 2*a;
    if(t1 > t_max || t2 < t_min)
        return false;

    t = (t1 > t_min? t1:t2);
    return true;
}
/**
*
*
*
*
*/

__device__
float schlick(float cos, float ref_idx)
{   
//real glass has reflectivity that varies with angle
//simple polynomial approximation by Christophe Schlick
    float r0 = (1.0f - ref_idx) / ( 1.0f + ref_idx );
    r0 = r0*r0;
    return r0 + (1-r0)*powf(1-cos, 5);
}

__device__
FVec3 reflect(const FVec3& vec, const FVec3& normal)
{
    return vec - normal*2*fvec3_dot(vec, normal);
}

__device__
bool refract(const FVec3& vec, const FVec3& normal, float ni_over_nt, FVec3& refracted)
{
    FVec3 unit_vec = fvec3_normalize(vec);
    float dot = fvec3_dot(unit_vec, normal);
    float discriminant = 1.0f - ni_over_nt*ni_over_nt *( 1.0f - dot*dot );
    if(discriminant > 0)
    {
        //refracted = normal*(ni_over_nt*dot - sqrtf(discriminant)) - unit_vec* ni_over_nt;
        refracted = (unit_vec-normal*dot)*ni_over_nt - normal*sqrtf(discriminant);
        return true;
    }
    return false;
}
/**
*
*
*
*
*/
enum material_type
{
    MT_LAMBERTIAN,
    MT_METAL,
    MT_DIELECTRIC,
};

struct material
{
    material_type type;
    FVec3 albedo;
    union
    {
        float fuzz;
        float ref_idx;
        
    } value;
};

__device__ bool scatter_ray(const material& mat, 
const h_ray::ray& ray, const FVec3& surface_normal,
curandState* local_rand_state,
FVec3& scattered_raydir)
{
    switch(mat.type)
    {
        case MT_LAMBERTIAN:
        {
            scattered_raydir = fvec3_normalize(
            surface_normal + random_in_unit_sphere(local_rand_state));
            return true;
        }break;
        
        case MT_METAL:
        {
            FVec3 reflection = reflect(ray.dir, surface_normal);
            scattered_raydir = reflection + 
            random_in_unit_sphere(local_rand_state) * mat.value.fuzz;
            return (fvec3_dot(scattered_raydir, surface_normal) > 0);
        }break;
        
        case MT_DIELECTRIC:
        {
            FVec3 reflection = reflect(ray.dir, surface_normal);
            float ni_over_nt = 0;
            FVec3 outward_normal;
            float cos = 0;

            float dot = fvec3_dot(ray.dir, surface_normal);
            if(dot > 0)
            {
                outward_normal = surface_normal * -1;
                ni_over_nt = mat.value.ref_idx;
                cos = mat.value.ref_idx * dot;
            }
            else
            {
                outward_normal = surface_normal;
                ni_over_nt = 1.0f / mat.value.ref_idx;
                cos = -dot;
            }
            
            FVec3 refraction;
            float reflection_prop = 1.0f;
            if(refract(ray.dir, outward_normal, ni_over_nt, refraction))
            {
                reflection_prop = schlick(cos, mat.value.ref_idx);
                scattered_raydir = refraction;
            }
            
            if( curand_uniform(local_rand_state) < reflection_prop )
            {
                scattered_raydir = reflection;
            }
            return true;
        }break;
    }
    
    return false;
}

__device__ material create_lambertian(const FVec3& albedo)
{
    material mat;
    mat.type = MT_LAMBERTIAN;
    mat.albedo = albedo;
    return mat;
}

__device__ material create_metal(const FVec3& albedo, float fuzz)
{
    material mat;
    mat.type = MT_METAL;
    mat.albedo = albedo;
    mat.value.fuzz = fuzz;
    return mat;
}

__device__ material create_dielectric(float ref_idx)
{
    material mat;
    mat.type = MT_DIELECTRIC;
    mat.albedo = {1.0f,1.0f,1.0f};
    mat.value.ref_idx = ref_idx;
    return mat;
}

/**
*
*
*
*
*/
enum shape_type 
{
    ST_SPHERE,
};

struct shape
{
    shape_type type;
    material mat;
    union
    {
        struct
        {
            FVec3 center;
            float radius;
        } sphere;
    } value;
};

__device__ shape create_sphere(const material& mat, const FVec3& center, float radius)
{
    shape sphere;
    sphere.type = ST_SPHERE;
    sphere.mat  = mat;
    sphere.value.sphere.center = center;
    sphere.value.sphere.radius = radius;
    return sphere;
}

__device__ FVec3 get_normal_at(const shape& shape, const FVec3& point)
{
    switch(shape.type)
    {
        case ST_SPHERE:
        {
            return ( point - shape.value.sphere.center ) / shape.value.sphere.radius;
        }break;
    }
    
    return {0,0,0};
}

__device__ bool shape_ray_hit(const shape& shape, const h_ray::ray& ray, 
float t_min, float t_max, float& t)
{
    switch(shape.type)
    {
        case ST_SPHERE:
        {
            return ray_sphere(
            ray, shape.value.sphere.center, shape.value.sphere.radius,
            t_min, t_max, t);
        }break;
    }
    
    return false;
}  



/**
*
*
*
*
*/
struct hit_record
{
  float t;
  shape* shape;
};

__device__ 
bool ray_shapes(const h_ray::ray& ray, shape shapes[], int num_shapes,
float t_min, float t_max, hit_record& record) 
{
// Check if a ray intersects any objects in the world,
// returns true and record written with t and shape intersected by the ray if hit, 
// otherwise return false
    bool hit = false;
    float closest_t = t_max;
    float temp = 0;
    for(int i = 0; i < num_shapes; ++i)
    {
        if(shape_ray_hit(shapes[i], ray, t_min, closest_t, temp))
        {
            closest_t = temp;
            record.t = closest_t;
            record.shape = shapes+i;
            hit = true;
        }
    }
    return hit;
}



__device__ 
FVec3 to_color(const h_ray::ray& init_ray, shape shapes[], int num_shapes,
                 int max_depth, curandState* local_rand_state)
{
// Path tracing
    FVec3 white = {1.0f, 1.0f, 1.0f};    
    FVec3 black = {0, 0, 0};
    
    hit_record record;
    h_ray::ray ray = init_ray;
    
    FVec3 attenuation = white;
          
    for(int depth = 0; depth < max_depth; ++depth)
    {
        if( !ray_shapes(ray, shapes, num_shapes, 0.001f, 100000.f, record) ) 
        {            
            float t = 0.5f * (ray.dir.y + 1.0f);          
            FVec3 ambient = {0.5f, 0.7f, 1.0f};
            FVec3 color = white*(1.0f - t) + ambient*t;
            return color * attenuation;
        }
        
        FVec3 point = h_ray::point_at_parameter(ray, record.t);        
        FVec3 normal = get_normal_at( *record.shape, point );
        FVec3 scattered_raydir;
        
        if( !scatter_ray( record.shape->mat, ray, normal,
        local_rand_state, scattered_raydir)) {
            return black * attenuation;
        }
        
        ray = h_ray::create_ray(point, scattered_raydir);
        attenuation *= record.shape->mat.albedo;
    }
       
    return black;
}

struct camera
{    
    FVec3 pos;
    float aspect_ratio;
    float vfov;
    
    // private
    FVec3 lower_left_corner;
    FVec3 horizontal;
    FVec3 vertical;
    float h;
    float w; 
};

__device__ float to_radians(float deg)
{
    return deg * 3.14159265359f / 180.f;
}

__device__
camera create_camera(const FVec3& pos, float aspect_ratio, float vfovDeg)
{
    camera cam;
    
    cam.pos = pos;
    cam.aspect_ratio = aspect_ratio;
    cam.vfov = to_radians(vfovDeg);
    float half_h = tan(cam.vfov/2);
    float half_w = aspect_ratio * half_h;
    
    cam.w = half_w * 2;
    cam.h = half_h * 2;
    
    cam.lower_left_corner = {-half_w, -half_h, -1};
    cam.horizontal = {cam.w, 0, 0};
    cam.vertical= {0, cam.h, 0};  
    return cam;
}

__device__
h_ray::ray get_ray_at(const camera& cam, float u, float v)
{
    return h_ray::create_ray( cam.pos, 
    cam.lower_left_corner + cam.horizontal * u + cam.vertical * v - cam.pos );
}


__global__
void render(unsigned char* fb, int max_x, int max_y, int ns,
camera** camera, shape** world, int* size, int max_depth,
curandState* rand_state)
{
// Render a pixel by casting a ray from camera to the point on viewport 
// computed by pixel position and framebuffer size
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

        h_ray::ray ray = get_ray_at(**camera, u, v);
        
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
void create_world(camera** cam, float aspect_ratio, shape** world, int* obj_size)
{
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        *cam = (camera*)malloc(sizeof(camera));
        **cam = create_camera( {0,0,0}, aspect_ratio, 90);
        
        *obj_size = 5;
        *world = (shape*)malloc(sizeof(shape) * (*obj_size));
        
        shape* s = *world;
        s[0] = create_sphere(create_lambertian({0.8f, 0.3f, 0.3f}),
                            {0, 0, -1}, 0.5f);
        s[1] = create_sphere(create_lambertian({0.8f, 0.8f, 0.0f}),
                            {0, -100.5f, -1}, 100.f);
        s[2] = create_sphere(create_metal({0.8f, 0.6f, 0.2f}, 0.0f),
                            {1, 0, -1}, 0.5f);
        s[3] = create_sphere(create_dielectric(1.5f),
                            {-1, 0, -1}, 0.5f);
        s[4] = create_sphere(create_dielectric(1.5f),
                            {-1, 0, -1}, -0.49f);
    }
}

__global__
void free_world(shape** world, camera** camera)
{
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        free(*world);
        free(*camera);
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
    shape** dev_world;
    int* dev_num_obj;
    camera** dev_cam;
    checkCudaError( cudaMalloc(&dev_world, sizeof(shape**)) );
    checkCudaError( cudaMalloc(&dev_num_obj, sizeof(int)) );
    checkCudaError( cudaMalloc(&dev_cam, sizeof(camera**)) );
  
    // Initialize objects in the world
    create_world<<<1,1>>>(dev_cam, float(ny)/nx, dev_world, dev_num_obj);
    checkCudaError( cudaGetLastError() );
    checkCudaError( cudaDeviceSynchronize() );
    
    
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
                                 dev_cam, dev_world, dev_num_obj, max_depth,
                                 dev_rand_state);
    checkCudaError( cudaGetLastError() );
    checkCudaError( cudaDeviceSynchronize() );
  
  
    // Deallocate objs in world
    free_world<<<1,1>>>(dev_world, dev_cam);
    checkCudaError( cudaGetLastError() );
    checkCudaError( cudaDeviceSynchronize() );
      
      
    // Deallocate heap world variables
    checkCudaError( cudaFree(dev_world) );
    checkCudaError( cudaFree(dev_num_obj) );
    checkCudaError( cudaFree(dev_cam) );
    
    // Deallocate random states
    checkCudaError( cudaFree(dev_rand_state) );
    
    print_ppm(fb, nx, ny);
    
    // Deallocate framebuffer
    checkCudaError( cudaFree(fb) );

    return 0;  
}
