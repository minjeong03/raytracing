#include <stdio.h>
#define HIMATH_IMPL
#include <vector>
#include <stdlib.h>
#include <time.h>
#include "himath.h"
#include <limits.h>
#include <process.h>
#include <windows.h>

#define NUM_THREADS 8
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

namespace h_shape{
  struct sphere
  {
    FVec3 center;
    float radius;
  };
}

namespace h_rayhit{
  struct hit_record
  {
    float t;
    FVec3 p;
    FVec3 normal;
  };
  
  typedef bool test_hit_fuc(void* obj, const h_ray::ray&, float t_min, float t_max, hit_record& record);
  struct hitable
  {
    test_hit_fuc* hit_func;
    void* hitable_obj;
  };

  hitable create_hitable(void* hitable_object, test_hit_fuc* hit_test_function)
  {
    hitable obj;
    obj.hitable_obj = hitable_object;
    obj.hit_func = hit_test_function;
    return obj;
  }

  bool test_world_ray(/*std::vector<hitable>*/void* world, const h_ray::ray& ray, float t_min, float t_max, hit_record& record)
  {
    if( !world ) return false;
    const std::vector<hitable>& objs = *((std::vector<hitable>*)(world));

    hit_record temp;
    bool hit_anything = false;
    double closest_t = t_max;

    for( const hitable& h : objs )
    {
      if(h.hit_func(h.hitable_obj, ray, t_min, t_max, temp))
      {
        hit_anything = true;
        if(closest_t > temp.t)
        {
          closest_t = temp.t;
          record = temp;
          record.t = closest_t;
        }
      }
    }
    return hit_anything;
  }

  bool test_sphere_ray(/*sphere**/void* obj, 
  const h_ray::ray& ray, float t_min, float t_max, hit_record& record)
  {
    if( !obj ) return false;
    const h_shape::sphere& sphere = *(h_shape::sphere*)(obj);

    FVec3 oc = ray.origin - sphere.center;
    float a = fvec3_dot(ray.dir, ray.dir);
    float b = 2 * fvec3_dot(ray.dir, oc);
    float c = fvec3_dot(oc, oc) - sphere.radius*sphere.radius;

    // use quadratic formula to test p = (origin + dir*t) and sphere for all t
    float discriminant = b*b - 4*a*c;
    if(discriminant < 0) 
      return false;
    
    // intersect
    else 
    {
      float sqrt_d = sqrt(discriminant);
      float t1 = ( - b - sqrt_d ) / (2*a);
      float t2 = ( - b + sqrt_d ) / (2*a);
      if( t1 > t_max || t2 < t_min )
        return false;

      record.t = (t1 > t_min ? t1 : t2);
      record.p = h_ray::point_at_parameter(ray, record.t);        
      record.normal = fvec3_normalize( record.p - sphere.center );
      return true;
    }
    return false;
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
  inline FVec3 color_gamma_2(FVec3 color)
  {
    return {sqrt(color.x), sqrt(color.y), sqrt(color.z)};
  }

  FVec3 to_color(const h_ray::ray& ray, const h_rayhit::hitable& world)
  {
    FVec3 white = {1.0f, 1.0f, 1.0f};
    FVec3 black = {0.0f, 0.0f, 0.0f};
   
    h_rayhit::hit_record record;
    if( world.hit_func(world.hitable_obj, ray, 0.001, FLT_MAX, record) )
    {
      h_ray::ray diffused_ray = h_ray::create_ray( record.p, fvec3_normalize(record.normal + h_random::random_in_unit_sphere()));
      return to_color(diffused_ray, world) * 0.5f;
    }
  
    float t = 0.5f * (ray.dir.y + 1.0f);
    FVec3 base = {0.5f, 0.7f, 1.0f};
    return  white*(1.0f - t) + base*t;
  }

  struct app_data
  {
    h_camera::camera camera;
    h_rayhit::hitable world;
    int w;
    int h;
    int bytesPerPixel;
    unsigned char* framebuffer;
    HANDLE thread_handles[NUM_THREADS];
  };
}

h_app::app_data gAppData;

struct thread_data
{
  int x0;
  int x1;
  int y0;
  int y1;
};

void write_buffer(int x0, int y0, int x1, int y1)
{
  const int sample = 100;  
  
  unsigned char* fb = gAppData.framebuffer;
  for (int j = y0; j < y1; j++) {
    for(int i = x0; i < x1; i++)   {
      FVec3 color = {0,0,0}; 

      for(int s = 0; s < sample; ++s) {
        float u = ( float(i) + h_random::randomf() ) / float(gAppData.w);
        float v = ( float(j) + h_random::randomf() ) / float(gAppData.h);
        
        h_ray::ray ray = h_camera::get_ray(gAppData.camera, u, v);
        color += h_app::to_color(ray, gAppData.world);
      }

      int index = (j * gAppData.w * gAppData.bytesPerPixel) + (i * gAppData.bytesPerPixel);

      color /= float(sample);
      color = h_app::color_gamma_2(color);
      fb[index + 0] = unsigned char(255.99f*color.x);
      fb[index + 1] = unsigned char(255.99f*color.y);
      fb[index + 2] = unsigned char(255.99f*color.z);
    }
  }
}

unsigned thread_main(void* args)
{
  thread_data* data = (thread_data*)args;

  write_buffer(data->x0, data->y0, data->x1, data->y1);

  free(data);
  _endthreadex(0);
  return 0;
}

// right handed coordinate
// +x-right
// +y-up
// +z-out of screen

void print_ppm()
{
  printf("P3\n%i %i\n255\n", gAppData.w, gAppData.h);

  unsigned char* fb = gAppData.framebuffer;
  for (int j = gAppData.h-1; j >= 0; j--) {
    for(int i = 0; i < gAppData.w; i++)   {
      int index = (j * gAppData.w * gAppData.bytesPerPixel ) + (i* gAppData.bytesPerPixel);
      unsigned char ir = fb[index + 0];
      unsigned char ig = fb[index + 1];
      unsigned char ib = fb[index + 2];
      printf("%i %i %i\n", ir, ig, ib);
    }
  }
}

bool create_threads()
{
  // TODO(minjeong): check if that W/H is dividable without remainders.
  int numH = 2;
  int numW = (NUM_THREADS >> 1);

  // TODO(minjeong): test with diffrent methods to dividing section.
  for(int i = 0; i < numH; ++i)
  {
    for(int j = 0; j < numW; ++j)
    {
      // TODO(minjeong): can reduce arg to int(section number or whatever).
      thread_data* data = (thread_data*)malloc(sizeof(thread_data));
      data->x0 = (gAppData.w/numW) * j; 
      data->y0 = (gAppData.h/numH) * i;
      data->x1 = (gAppData.w/numW) * (j+1); 
      data->y1 = (gAppData.h/numH) * (i+1);

      gAppData.thread_handles[i*numW + j] = (HANDLE)_beginthreadex( NULL, 0, thread_main, data, 0, NULL);
      if( !gAppData.thread_handles[i*numW + j] )
        return false;
    }
  }
  return true;
}

int main() {
  srand(time(NULL));

  gAppData.w = 1600;
  gAppData.h = 900;
  gAppData.bytesPerPixel = 3;

  gAppData.camera = h_camera::create_camera({0,0,0});
  
  h_shape::sphere obj1;
  obj1.center = {0,0,-1};
  obj1.radius = 0.5f;
  h_shape::sphere obj2;
  obj2.center = {0, -100.5, -1};
  obj2.radius = 100;
  std::vector<h_rayhit::hitable> objs_in_world;
  objs_in_world.push_back( h_rayhit::create_hitable(&obj1, h_rayhit::test_sphere_ray) );
  objs_in_world.push_back( h_rayhit::create_hitable(&obj2, h_rayhit::test_sphere_ray) );

  gAppData.world = h_rayhit::create_hitable(&objs_in_world, h_rayhit::test_world_ray);
  gAppData.framebuffer = (unsigned char*)malloc(gAppData.w * gAppData.h * sizeof(unsigned char)* gAppData.bytesPerPixel);

  if(create_threads())
  { 
    fprintf(stderr, "Writing buffer using multi threads..");
    DWORD wait_result = WaitForMultipleObjects( NUM_THREADS, gAppData.thread_handles, TRUE, INFINITE);
    if(wait_result != WAIT_OBJECT_0)
    {
      fprintf(stderr, "Failed waiting threads %i\n", wait_result);
      fprintf(stderr, "failed with %i", GetLastError());
      return -1;
    }

    for(int i = 0; i < NUM_THREADS; ++i)
      CloseHandle(gAppData.thread_handles);
  }
  else
  {
    fprintf(stderr, "Writing buffer without multi threads..");
    write_buffer(0, 0, gAppData.w, gAppData.h);
  }

  print_ppm();
  free(gAppData.framebuffer);
  return 0;  
}
