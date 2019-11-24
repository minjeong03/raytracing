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

int main() {
  int nx = 200;
  int ny = 100;
  printf("P3\n%i %i\n255\n", nx, ny);
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
