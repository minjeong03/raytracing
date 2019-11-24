#include <stdio.h>
#define HIMATH_IMPL
#include "himath.h"

namespace h_ray{
  struct ray
  {
    FVec3 origin;
    FVec3 dir;
  };

  FVec3 point_at_parameter(float t, const ray& ray)
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

  bool sphere_test_inside(const sphere& sphere, FVec3 point)
  {
    FVec3 pc = point - sphere.center;
    return (fvec3_dot(pc,pc) <= sphere.radius*sphere.radius);
  }
}

namespace h_intersect{
  struct ray_intersect_result
  {
    float t;
    bool hit;
    bool inside;
  };

  bool intersect_ray_sphere(const h_ray::ray& ray, const h_shape::sphere& sphere)
  {
    FVec3 oc = ray.origin - sphere.center;
    float a = fvec3_dot(ray.dir, ray.dir);
    float b = 2 * fvec3_dot(ray.dir, oc);
    float c = fvec3_dot(oc, oc) - sphere.radius*sphere.radius;

    // use quadratic formula to test p = (origin + dir*t) and sphere for all t
    float discriminant = b*b - 4*a*c;
    return (discriminant > 0);

    /*
      float t = 0; 
      // ray and sphere 
      // don't intersect
      if(discriminant < 0)
      {
      
      }
      // intersect at a point
      else if(discriminant == 0)
      {

      }
      // intersect at two points
      else
      {

      }
    */
  }
}

namespace h_app{
  FVec3 to_color(const h_ray::ray& ray)
  {
    FVec3 unit_dir = fvec3_normalize(ray.dir);
    float t = 0.5f * (unit_dir.y + 1.0f); // map y [-1, 1] to [0, 1]
    
    h_shape::sphere sphere;
    sphere.center = {0,0,-1};
    sphere.radius = 0.5f;
    
    if(h_intersect::intersect_ray_sphere(ray, sphere))
      return {1,0,0};
            
    FVec3 white = {1.0f, 1.0f, 1.0f};
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

  // r = (1, 0]
  // g = [0, 1)
  // b = 0.2;
  for (int j = ny-1; j >= 0; j--) {
    for(int i = 0; i < nx; i++)   {
      float u = float(i) / float(nx);
      float v = float(j) / float(ny);

      // dir's transition = from top-left to bottom-right
      ray.dir = lower_left_corner + horizontal * u + vertical * v;
      FVec3 color = h_app::to_color(ray);

      int ir = int(255.99f*color.x);
      int ig = int(255.99f*color.y);
      int ib = int(255.99f*color.z);

      printf("%i %i %i\n", ir, ig, ib);
    }
  }

  return 0;  
}
