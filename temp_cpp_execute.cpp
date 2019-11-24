#include <stdio.h>
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

FVec3 to_color(const h_ray::ray& ray)
{
  FVec3 unit_dir = fvec3_normalize(ray.dir);
  float t = 0.5f * (unit_dir.y + 1.0f); // map y [-1, 1] to [0, 1]
  
  FVec3 white = {1.0f, 1.0f, 1.0f};
  FVec3 base = {0.5f, 0.7f, 1.0f};

  return  white*(1.0f - t) + base*t;
}

int main() {
  int nx = 200;
  int ny = 100;
  printf("P3\n%i %i\n255\n", nx, ny);

  // r = (1, 0]
  // g = [0, 1)
  // b = 0.2;
  for (int j = ny-1; j >= 0; j--) {
    for(int i = 0; i < nx; i++)   {
      FVec3 color = {float(i) / float(nx), float(j) / float(ny), 0.2f};

      int ir = int(255.99f*color.x);
      int ig = int(255.99f*color.y);
      int ib = int(255.99f*color.z);

      printf("%i %i %i\n", ir, ig, ib);
    }
  }

  return 0;  
}
