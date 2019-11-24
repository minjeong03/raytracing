#include <stdio.h>
#define HIMATH_IMPL
#include "himath.h"
#include <vector>

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
    std::vector<hitable>& objs = *((std::vector<hitable>*)(world));

    hit_record temp;
    bool hit_anything = false;
    double closest_t = t_max;

    for( hitable& h : objs )
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
    h_shape::sphere& sphere = *(h_shape::sphere*)(obj);

    FVec3 oc = ray.origin - sphere.center;
    float a = fvec3_dot(ray.dir, ray.dir);
    float b = 2 * fvec3_dot(ray.dir, oc);
    float c = fvec3_dot(oc, oc) - sphere.radius*sphere.radius;

    // use quadratic formula to test p = (origin + dir*t) and sphere for all t
    float discriminant = b*b - 4*a*c;

      // ray and sphere 
      // don't intersect
      if(discriminant < 0) 
        return false;
      
      // intersect
      // sqrt_d == 0 -> only one  
      // sqrt_d > 0
      //      t1 | t2 | hit    
      // min  -  |  - | false      ==> meaning t1 < t2 < min
      // min  +  |  - | impossible ==> meaning t2 < min < t1 
      // max  -  |  + | true       ==> meaning t1 < min < t2
      // max  +  |  + | false      ==> meaning max < t1 < t2 
      else 
      {
        float sqrt_d = sqrt(discriminant);
        float t1 = ( - b - sqrt_d ) / (2*a);
        float t2 = ( - b + sqrt_d ) / (2*a);
        if( t1 > t_max || t2 < t_min )
          return false;

        record.t = (t1 < t_min ? t2 : t1);
        record.p = h_ray::point_at_parameter(ray, record.t);        
        record.normal = fvec3_normalize( record.p - sphere.center );
        return true;
      }
      return false;
  }
}

namespace h_app{
  FVec3 to_color(const h_ray::ray& ray, h_rayhit::hitable* world)
  {
    FVec3 white = {1.0f, 1.0f, 1.0f};
    FVec3 black = {0.0f, 0.0f, 0.0f};
    
    float min_t = 0;
    float max_t = 10000;
    h_rayhit::hit_record record;
    if( world->hit_func(world->hitable_obj, ray, min_t, max_t, record) )
    {
      FVec3 normal = (record.normal + white) * 0.5f;
      return normal;
    }

    float t = 0.5f * (ray.dir.y + 1.0f);

    FVec3 base = {0.5f, 0.7f, 1.0f};

    return  white*(1.0f - t) + base*t;
  }
}


// right handed coordinate
// +x-right
// +y-up
// +z-out of screen
int main() {
  int nx = 200;
  int ny = 100;
  printf("P3\n%i %i\n255\n", nx, ny);

  FVec3 lower_left_corner = {-2, -1, -1};
  FVec3 horizontal = {4, 0, 0};
  FVec3 vertical = {0, 2, 0};
  FVec3 camera_pos = {0,0,0};
  
  h_ray::ray ray;
  ray.origin = camera_pos;

  h_shape::sphere obj1;
  obj1.center = {0,0,-1};
  obj1.radius = 0.5f;
  h_shape::sphere obj2;
  obj2.center = {0, -100.5, -1};
  obj2.radius = 100;
  std::vector<h_rayhit::hitable> objs_in_world;
  objs_in_world.push_back( h_rayhit::create_hitable(&obj1, h_rayhit::test_sphere_ray) );
  objs_in_world.push_back( h_rayhit::create_hitable(&obj2, h_rayhit::test_sphere_ray) );

  h_rayhit::hitable world = h_rayhit::create_hitable(&objs_in_world, h_rayhit::test_world_ray);

  // u = (1, 0]
  // v = (0, 1]
  for (int j = ny-1; j >= 0; j--) {
    for(int i = 0; i < nx; i++)   {
      float u = float(i) / float(nx);
      float v = float(j) / float(ny);

      // dir's transition = from top-left to bottom-right
      ray.dir = lower_left_corner + horizontal * u + vertical * v;
      FVec3 color = h_app::to_color(ray, &world);

      int ir = int(255.99f*color.x);
      int ig = int(255.99f*color.y);
      int ib = int(255.99f*color.z);

      printf("%i %i %i\n", ir, ig, ib);
    }
  }

  return 0;  
}
