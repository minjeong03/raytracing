#include <stdio.h>
#define HIMATH_IMPL
#include <vector>
#include <stdlib.h>
#include <time.h>
#include "himath.h"
#include <limits.h>
#include <process.h>
#include <windows.h>

#define checkCudaError(val) check_cuda( (val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, 
char const *const func, const char* const file, int const line)
{
    if(result) {
        fprintf(stderr, "CUDA error = %i at %s : %i '%s'\n", unsigned(result),
        file, line, func);
        
        cudaDeviceReset();
        exit(99);
    }
}

namespace h_random {
  float randomf(void)
  {
    float v = rand();
    float m = RAND_MAX + 1;
    
    // return [0,1)
    return v / m;   
  }

  FVec3 random_in_unit_sphere()
  {
    FVec3 rand = {0,0,0};
    // [0,1) => [0, 2) => [-1, 1)
    do
    {
      rand.x = 2*randomf() -1.f;
      rand.y = 2*randomf() -1.f;
      rand.z = 2*randomf() -1.f;   
    }
    while(fvec3_dot(rand, rand) >= 1.f);
    return rand;  
  }
}

namespace h_ray{
  struct ray
  {
    FVec3 origin;
    FVec3 dir;
  };

  FVec3 point_at_parameter(const ray& ray, float t)
  {
    FVec3 p = ray.origin + ray.dir*t;
    return p;
  }

  ray create_ray(FVec3 origin, FVec3 dir)
  {
    ray r;
    r.origin = origin;
    r.dir = dir;
    return r;
  }
}

namespace h_util {

  //real glass has reflectivity that varies with angle
  //simple polynomial approximation by Christophe Schlick
  float schlick(float cos, float ref_idx) {
    float r0 = (1.0f - ref_idx) / ( 1.0f + ref_idx);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1-cos), 5);
  }
  
  inline FVec3 color_gamma_2(FVec3 color)
  {
    return {sqrt(color.x), sqrt(color.y), sqrt(color.z)};
  }

}

namespace h_vector{
  FVec3 reflect(const FVec3& v, const FVec3& n)
  {
    return v - n * 2 * fvec3_dot(v, n);
  }
  bool refract(const FVec3& v, const FVec3& n, float ni_over_nt, FVec3* pRefracted)
  {
    FVec3 uv = fvec3_normalize(v);
    float dt = fvec3_dot(uv, n);
    float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
    if(pRefracted && discriminant > 0)
    {
      *pRefracted = (uv - n*dt)*ni_over_nt - n*sqrt(discriminant);
      return true;
    }
    return false;
  }
}


namespace h_shape{
  struct sphere
  {
    FVec3 center;
    float radius;
  };

  sphere create_sphere(FVec3 center, float radius)
  {
    sphere sphere;
    sphere.center = center;
    sphere.radius = radius;
    return sphere;
  }
}

namespace h_material{
  struct lambertian
  {
    FVec3 albedo;
  };

  lambertian create_lambertian(FVec3 albedo)
  {
    lambertian a;
    a.albedo = albedo;
    return a;
  }
  
  struct metal
  {
    FVec3 albedo;
    float fuzz;
  };
  
  metal create_metal(FVec3 albedo, float fuzz)
  {
    metal a;
    a.albedo = albedo;
    a.fuzz = fuzz;
    return a;
  }
  
  struct dielectric
  {
    float refract_index;
  };

  dielectric create_dielectric(float ref_ind)
  {
    dielectric a;
    a.refract_index = ref_ind;
    return a;
  }
}

namespace h_rayhit{
  
  
  struct hit_record
  {
    float t;
    FVec3 p;
    FVec3 normal;
    bool hit;
  };

    struct scatter_result
  {
    h_ray::ray ray;
    FVec3 attenuation;
    bool scattered;
  };

  typedef hit_record test_hit_fuc(void* obj, const h_ray::ray&, float t_min, float t_max);
  typedef scatter_result scatter_func(void* material, const h_ray::ray& ray, const hit_record& record);

  struct hitable
  {
    void* shape;
    test_hit_fuc* hit_func;

    void* material;
    scatter_func* scatter_func;
  };

  hitable create_hitable(void* shape, test_hit_fuc* hit_test_function, void* material, scatter_func* scatter_func)
  {
    hitable obj;
    obj.shape = shape;
    obj.material = material;
    obj.hit_func = hit_test_function;
    obj.scatter_func = scatter_func;
    return obj;
  }

  hit_record test_sphere_ray(/*sphere**/void* obj, 
  const h_ray::ray& ray, float t_min, float t_max)
  {
    hit_record record;
    record.hit = false;
    if( !obj ) return record;
    const h_shape::sphere& sphere = *(h_shape::sphere*)(obj);

    FVec3 oc = ray.origin - sphere.center;
    float a = fvec3_dot(ray.dir, ray.dir);
    float b = 2 * fvec3_dot(ray.dir, oc);
    float c = fvec3_dot(oc, oc) - sphere.radius*sphere.radius;

    float discriminant = b*b - 4*a*c;

      if(discriminant < 0) 
        return record;
      else 
      {
        float sqrt_d = sqrt(discriminant);
        float t1 = ( - b - sqrt_d ) / (2*a);
        float t2 = ( - b + sqrt_d ) / (2*a);
        if( t1 > t_max || t2 < t_min )
          return record;

        record.hit = true;
        record.t = (t1 > t_min ? t1 : t2);
        record.p = h_ray::point_at_parameter(ray, record.t);        
        record.normal = ( record.p - sphere.center ) / sphere.radius;
        return record;
      }
      return record;
  }

  
  scatter_result lambertian_scatter(void* mat, const h_ray::ray& ray, const hit_record& record)
  {
    scatter_result result;
    const h_material::lambertian& material = *((h_material::lambertian*)mat);
    result.ray = h_ray::create_ray( record.p, fvec3_normalize(record.normal + h_random::random_in_unit_sphere()));
    result.attenuation = material.albedo;
    result.scattered = true;
    return result;
  }
  scatter_result metal_scatter(void* mat, const h_ray::ray& ray, const hit_record& record)
  {
    scatter_result result;
    const h_material::metal& material = *((h_material::metal*)mat);
    float fuzz = (material.fuzz > 1.0f ? 1.0f: material.fuzz );
    FVec3 reflected = h_vector::reflect(fvec3_normalize(ray.dir), record.normal);
    result.ray = h_ray::create_ray( record.p, reflected + h_random::random_in_unit_sphere()* fuzz);
    result.attenuation = material.albedo;
    result.scattered = (fvec3_dot(result.ray.dir, record.normal) > 0);
    return result;
  }
  scatter_result dielectric_scatter(void* mat, const h_ray::ray& ray, const hit_record& record)
  {
    scatter_result result;
    const h_material::dielectric& material = *((h_material::dielectric*)mat);
    FVec3 reflected = h_vector::reflect(ray.dir, record.normal);
    result.attenuation = {1.0f, 1.0f, 1.0f};

    float ni_over_nt = 0;
    FVec3 outward_normal;
    float cos = 0;
    
    if(fvec3_dot(ray.dir, record.normal) > 0) {
      outward_normal = record.normal*-1;
      ni_over_nt = material.refract_index;
      cos = material.refract_index * fvec3_dot(ray.dir, record.normal) ;
    }
    else {
      outward_normal = record.normal;
      ni_over_nt = 1.0f / material.refract_index;
      cos = -fvec3_dot(ray.dir, record.normal);
    }

    FVec3 refracted;
    float reflect_prop;
    if(h_vector::refract(ray.dir, outward_normal, ni_over_nt, &refracted))
    {
      reflect_prop = h_util::schlick(cos, material.refract_index);
      result.ray = h_ray::create_ray(record.p, refracted);  
    } 
    else
    {
      reflect_prop = 1.0f;
    }

    if(h_random::randomf() < reflect_prop)
    {
      result.ray = h_ray::create_ray(record.p, reflected);  
    }
    
    result.scattered = true;
    return result;
  }
}

namespace h_camera {
  struct camera
  {
    FVec3 lower_left_corner;
    FVec3 horizontal;
    FVec3 vertical;
    FVec3 pos;
  };

  camera create_camera(FVec3 pos) {
    camera cam;
    cam.lower_left_corner = {-2, -1, -1};
    cam.horizontal = {4, 0, 0};
    cam.vertical = {0, 2, 0};
    cam.pos = pos;
    return cam;
  }

  h_ray::ray get_ray(const camera& camera, float u, float v) {
    h_ray::ray ray; 
    ray.origin = camera.pos;
    ray.dir = camera.lower_left_corner + camera.horizontal*u + camera.vertical*v - camera.pos;
    return ray;
  }
}

namespace h_app{
  struct hit_result
  {
    h_rayhit::hit_record record;
    void* material;
    h_rayhit::scatter_func* scatter_func;
  };

  hit_result test_world_ray(h_rayhit::hitable hitable_array[], int size, 
                      const h_ray::ray& ray, float t_min, float t_max)
  {
    hit_result result;
    result.record.hit = false;
    if( !hitable_array ) return result;

    h_rayhit::hit_record temp;
    double closest_t = t_max;

    for(int i = 0; i < size; ++i)
    {
      const h_rayhit::hitable& h = hitable_array[i];
      temp = h.hit_func(h.shape, ray, t_min, t_max); 
      if(temp.hit)
      {
        if(closest_t > temp.t)
        {
          closest_t = temp.t;
          result.record = temp;
          result.material = h.material;
          result.scatter_func = h.scatter_func;
        }
      }
    }
    return result;
  }


  FVec3 to_color(const h_ray::ray& ray, h_rayhit::hitable world[], int size, int depth)
  {
    FVec3 white = {1.0f, 1.0f, 1.0f};
    FVec3 black = {0.0f, 0.0f, 0.0f};
   
    h_app::hit_result result = test_world_ray(world, size, ray, 0.001, FLT_MAX);
    if( result.record.hit )
    {
      h_rayhit::scatter_result scattered = result.scatter_func(result.material, ray, result.record);
      if(depth < 50 && scattered.scattered)
      {
        return to_color(scattered.ray, world, size, depth+1) * scattered.attenuation;
      }
      else
      {
        return black;
      }
    }
  
    float t = 0.5f * (ray.dir.y + 1.0f);
    FVec3 base = {0.5f, 0.7f, 1.0f};
    return  white*(1.0f - t) + base*t;
  }

  struct app_data
  {
    h_camera::camera camera;
    h_rayhit::hitable* world;
    int size;
    int w;
    int h;
    unsigned char* framebuffer;
  };
}


void write_buffer(int x0, int y0, int x1, int y1, unsigned char* fb, 
const h_app::app_data& gAppData)
{
  const int sample = 100;  
  for (int j = y0; j < y1; j++) {
    for(int i = x0; i < x1; i++)   {
      FVec3 color = {0,0,0}; 

      for(int s = 0; s < sample; ++s) {
        float u = ( float(i) + h_random::randomf() ) / float(gAppData.w);
        float v = ( float(j) + h_random::randomf() ) / float(gAppData.h);
        
        h_ray::ray ray = h_camera::get_ray(gAppData.camera, u, v);
        color += h_app::to_color(ray, gAppData.world, gAppData.size, 0);
      }

      int index = (j * gAppData.w * 3) + (i * 3);

      color /= float(sample);
      color = h_util::color_gamma_2(color);
      fb[index + 0] = unsigned char(255.99f*color.x);
      fb[index + 1] = unsigned char(255.99f*color.y);
      fb[index + 2] = unsigned char(255.99f*color.z);
    }
  }
}

// right handed coordinate
// +x-right
// +y-up
// +z-out of screen

void print_ppm(int w, int h, unsigned char* fb)
{
  printf("P3\n%i %i\n255\n", w, h);

  for (int j = h-1; j >= 0; j--) {
    for(int i = 0; i < w; i++)   {
      int index = (j * w * 3) + (i* 3);
      unsigned char ir = fb[index + 0];
      unsigned char ig = fb[index + 1];
      unsigned char ib = fb[index + 2];
      printf("%i %i %i\n", ir, ig, ib);
    }
  }
}


int main() {
  srand(time(NULL));
  
  h_app::app_data gAppData;
  
  gAppData.w = 800;
  gAppData.h = 400;

  gAppData.camera = h_camera::create_camera({0,0,0});
  
  h_shape::sphere sphere[5];
  sphere[0] = h_shape::create_sphere({0,0,-1}, 0.5f);
  sphere[1] = h_shape::create_sphere({0,-100.5,-1}, 100.f);
  sphere[2] = h_shape::create_sphere({1,0,-1}, 0.5f);
  sphere[3] = h_shape::create_sphere({-1,0,-1}, 0.5f);
  sphere[4] = h_shape::create_sphere({-1,0,-1}, -0.45f);

  h_material::lambertian sphere1Mat = h_material::create_lambertian({0.8f, 0.3f, 0.3f}); 
  h_material::lambertian sphere2Mat= h_material::create_lambertian({0.8f, 0.8f, 0.0f});
  h_material::metal sphere3Mat= h_material::create_metal({0.8f, 0.6f, 0.2f}, 0.0f);
  h_material::dielectric sphere4Mat = h_material::create_dielectric(1.5f);
  h_material::dielectric sphere5Mat = h_material::create_dielectric(1.5f);
    
  h_rayhit::hitable world[5];
  world[0] = h_rayhit::create_hitable(&sphere[0], h_rayhit::test_sphere_ray, &sphere1Mat, h_rayhit::lambertian_scatter);
  world[1] = h_rayhit::create_hitable(&sphere[1], h_rayhit::test_sphere_ray, &sphere2Mat, h_rayhit::lambertian_scatter);
  world[2] = h_rayhit::create_hitable(&sphere[2], h_rayhit::test_sphere_ray, &sphere3Mat, h_rayhit::metal_scatter);
  world[3] = h_rayhit::create_hitable(&sphere[3], h_rayhit::test_sphere_ray, &sphere4Mat, h_rayhit::dielectric_scatter);
  world[4] = h_rayhit::create_hitable(&sphere[4], h_rayhit::test_sphere_ray, &sphere5Mat, h_rayhit::dielectric_scatter);
  
  int size = sizeof(world)/sizeof(h_rayhit::hitable);

  gAppData.world = world;
  gAppData.size = size;
  gAppData.framebuffer = (unsigned char*)malloc(gAppData.w * gAppData.h * sizeof(unsigned char)* 3);

  write_buffer(0, 0, gAppData.w, gAppData.h, gAppData.framebuffer, gAppData);
  print_ppm(gAppData.w, gAppData.h, gAppData.framebuffer);
  
  free(gAppData.framebuffer);
  return 0;  
}
