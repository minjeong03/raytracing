#include <stdio.h>
#define HIMATH_IMPL
#include "himath.h"
#include <curand_kernel.h>
#include "tomlc99/toml.h"   


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

/*******************/ 
/*  random number  */
/*******************/
__device__ FVec2 random_in_unit_disc(curandState* local_rand_state)
{
    FVec2 rand = { 0,0 };
    do
    {
        rand.x = 2.f * curand_uniform(local_rand_state) - 1.f;
        rand.y = 2.f * curand_uniform(local_rand_state) - 1.f;
    }while(rand.x* rand.x + rand.y* rand.y >= 1.0f);
    
    return rand;
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


/*********/ 
/*  ray  */
/*********/
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

/***********************/ 
/*  ray intersections  */
/***********************/
__device__ bool 
ray_sphere(const ray& ray, const FVec3& center, float radius, 
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



/********** 
 *  math  *
 **********/
__device__
float schlick(float cos, float ref_idx)
{   
//real glass has reflectivity that varies with angle
//simple polynomial approximation by Christophe Schlick
    float r0 = (1.0f - ref_idx) / ( 1.0f + ref_idx );
    r0 = r0*r0;
    return r0 + (1-r0)*powf(1-cos, 5);
}

__device__ __host__ float to_radians(float deg)
{
    return deg * 3.14159265359f / 180.f;
}

/********************** 
 *  vector operation  *
 **********************/
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

/************** 
 *  material  *
 **************/
enum material_type
{
    MT_LAMBERTIAN,
    MT_METAL,
    MT_DIELECTRIC,
    MT_ERROR,
};

material_type string_to_material_type(const char* str)
{
    if(strcmp(str, "MT_LAMBERTIAN") == 0)
    {
      return material_type::MT_LAMBERTIAN;
    }
    else if(strcmp(str, "MT_METAL") == 0)
    {
      return material_type::MT_METAL;    
    }
    else if(strcmp(str, "MT_DIELECTRIC") == 0)
    {
      return material_type::MT_DIELECTRIC;
    }
    
    return material_type::MT_ERROR;
}
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
const ray& ray, const FVec3& surface_normal,
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

__device__ __host__ material create_lambertian(const FVec3& albedo)
{
    material mat;
    mat.type = MT_LAMBERTIAN;
    mat.albedo = albedo;
    return mat;
}

__device__ __host__ material create_metal(const FVec3& albedo, float fuzz)
{
    material mat;
    mat.type = MT_METAL;
    mat.albedo = albedo;
    mat.value.fuzz = fuzz;
    return mat;
}

__device__ __host__ material create_dielectric(float ref_idx)
{
    material mat;
    mat.type = MT_DIELECTRIC;
    mat.albedo = {1.0f,1.0f,1.0f};
    mat.value.ref_idx = ref_idx;
    return mat;
}

/***********
 *  shape  *
 **********/
enum shape_type 
{
    ST_SPHERE,
    ST_ERROR,
};

shape_type string_to_shape_type(const char* str)
{
    if(strcmp(str, "ST_SPHERE") == 0)
    {
      return shape_type::ST_SPHERE;
    }
    
    return shape_type::ST_ERROR;
}

struct sphere_shape
{
    FVec3 center;
    float radius;
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

__device__ bool ray_shape_hit(const shape& shape, const ray& ray, 
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


/********************
 *  shape - sphere  *
 ********************/
__device__ __host__ 
shape create_sphere(const material& mat, const FVec3& center, float radius)
{
    shape sphere;
    sphere.type = ST_SPHERE;
    sphere.mat  = mat;
    sphere.value.sphere.center = center;
    sphere.value.sphere.radius = radius;
    return sphere;
}


/****************
 *  raycasting  *
 ****************/
struct hit_record
{
  float t;
  shape* shape;
};

__device__ 
bool get_closest_ray_shape_hit(const ray& ray, shape shapes[], int num_shapes,
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
        if(ray_shape_hit(shapes[i], ray, t_min, closest_t, temp))
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
FVec3 ray_to_color(const ray& init_ray, shape shapes[], int num_shapes,
                 int max_depth, curandState* local_rand_state)
{
// Path tracing
    FVec3 white = {1.0f, 1.0f, 1.0f};    
    FVec3 black = {0, 0, 0};
    
    hit_record record;
    ray ray = init_ray;
    
    FVec3 attenuation = white;
          
    for(int depth = 0; depth < max_depth; ++depth)
    {
        if( !get_closest_ray_shape_hit( 
        ray, shapes, num_shapes, 0.001f, 100000.f, record ) ) 
        {            
            float t = 0.5f * (ray.dir.y + 1.0f);          
            FVec3 ambient = {0.5f, 0.7f, 1.0f};
            FVec3 color = white*(1.0f - t) + ambient*t;
            return color * attenuation;
        }
        
        FVec3 point = point_at_parameter(ray, record.t);        
        FVec3 normal = get_normal_at( *record.shape, point );
        FVec3 scattered_raydir;
        
        if( !scatter_ray( record.shape->mat, ray, normal,
        local_rand_state, scattered_raydir)) {
            return black * attenuation;
        }
        
        ray = create_ray(point, scattered_raydir);
        attenuation *= record.shape->mat.albedo;
    }
       
    return black;
}

/************
 *  camera  *
 ************/
struct camera
{    
    FVec3 pos;
    float aspect_ratio;
    float vfov;
    
    // private
    FVec3 lower_left_corner;
    FVec3 horizontal;
    FVec3 vertical;
    FVec3 up;
    FVec3 right;
    FVec3 view;
    
    float lens_radius;
};

__device__ __host__
camera create_camera(const FVec3& pos, const FVec3& lookAt, const FVec3& vup,
float aspect_ratio, float vfov_degree, float focus_dist, float aperture)
{
    camera cam;
    
    cam.pos = pos;
    cam.aspect_ratio = aspect_ratio;
    cam.vfov = to_radians(vfov_degree);
    float half_h = tan(cam.vfov/2);
    float half_w = aspect_ratio * half_h;
  
    // from pos to lookAt is view direction; which is -view
    cam.view = fvec3_normalize(pos - lookAt); 
    cam.right = fvec3_normalize(fvec3_cross(vup, cam.view));
    cam.up = fvec3_cross(cam.view, cam.right);
    
    cam.lower_left_corner = cam.pos - cam.right*half_w*focus_dist - cam.up*half_h*focus_dist -cam.view*focus_dist;
    cam.horizontal = cam.right*half_w*2*focus_dist;
    cam.vertical= cam.up*half_h*2*focus_dist;  
    cam.lens_radius= aperture/2;
    return cam; 
}

__device__
ray get_ray_at(const camera& cam, float u, float v, curandState* local_rand_state)
{
    FVec2 rand2d = random_in_unit_disc(local_rand_state);
    FVec3 offset = cam.right * (rand2d.x * cam.lens_radius) + cam.up* (rand2d.y * cam.lens_radius);
    
    return ::create_ray( cam.pos + offset, 
    cam.lower_left_corner + cam.horizontal * u + cam.vertical * v - (cam.pos + offset) );
}

/************
 *  render  *
 ************/
__global__
void render(unsigned char* fb, int max_x, int max_y, int ns,
camera* camera, shape* world, int size, int max_depth,
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

        ray ray = get_ray_at(*camera, u, v, local_rand_state);
        
        color += ray_to_color(ray, world, size, max_depth, local_rand_state);
    }
    color /= float(ns);
    color.x = sqrtf(color.x);
    color.y = sqrtf(color.y);
    color.z = sqrtf(color.z);
    
    fb[index*3+0] = unsigned char(255.99f * color.x);
    fb[index*3+1] = unsigned char(255.99f * color.y);
    fb[index*3+2] = unsigned char(255.99f * color.z);
}

/********************
 *  generate image  *
 ********************/
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
/********************
 ****  app data
 ********************/
struct app_data
{
  int nx;
  int ny;
  int ns;  
  
  int thread_x;
  int thread_y;
  
  int max_reflect;
  
  int num_objs;
  shape* shapes;
  
  camera camera;
};


/********************
 *** toml utilitis  *
 ********************/
bool my_toml_raw_to_int( toml_table_t* toml_table, const char* key, int& ret )
{
    const char* s_ret = toml_raw_in( toml_table, key );
    if( !s_ret )
    {
        fprintf(stderr, "TOML error = failed to read key( %s )\n", key);
        return false;
    }
    
    int64_t i_ret;
    if( toml_rtoi( s_ret, &i_ret ) )
    {
        fprintf(stderr, "TOML error = failed to convert %s to int\n", s_ret);
        return false;
    }
    
    ret = i_ret;
    return true;
}

bool my_toml_raw_to_float( toml_table_t* toml_table, const char* key, float& ret )
{
    const char* s_ret = toml_raw_in( toml_table, key );
    if( !s_ret )
    {
        fprintf(stderr, "TOML error = failed to read key( %s )\n", key);
        return false;
    }
    
    double d_ret;
    if( toml_rtod( s_ret, &d_ret ) )
    {
        fprintf(stderr, "TOML error = failed to convert %s to float\n", s_ret);
        return false;
    }
    
    ret = d_ret;
    return true;
}

bool my_toml_table_to_vector3( toml_table_t* toml_table, const char* key, FVec3& vec)
{
    bool success = true;
    toml_table_t* temp = toml_table_in( toml_table, key );
    if( !temp )
    {
        fprintf(stderr, "TOML error = failed to load vector3 %s\n", key);
        return false;
    }
    success &= my_toml_raw_to_float( temp, "x", vec.x );
    success &= my_toml_raw_to_float( temp, "y", vec.y );
    success &= my_toml_raw_to_float( temp, "z", vec.z );
    return success;
}

bool load_material( toml_table_t* parent_toml, material* ret )
{    
    toml_table_t* mat_toml = toml_table_in( parent_toml, "material");
    if( !mat_toml)
    {
        fprintf(stderr, "TOML error = failed to read 'material' in object\n");
        return false;
    }
    
    const char* mat_type_str = toml_raw_in( mat_toml, "type" );
    if( !mat_type_str )
    {
        fprintf(stderr, "TOML error = failed to read material type\n");
        return false;    
    }
            
    bool success = true;
    material_type type = string_to_material_type( mat_type_str );
    switch( type )
    {
        case MT_LAMBERTIAN:
        {
            FVec3 albedo;
            success &= my_toml_table_to_vector3( mat_toml, "albedo", albedo);
            *ret = create_lambertian( albedo );
        }break;
        case MT_METAL:
        {
            FVec3 albedo;
            float fuzz;
            success &= my_toml_table_to_vector3( mat_toml, "albedo", albedo);
            success &= my_toml_raw_to_float( mat_toml, "fuzz", fuzz);
            *ret = create_metal( albedo, fuzz );
        }break;
        case MT_DIELECTRIC:
        {
            float ref_idx;
            success &= my_toml_raw_to_float( mat_toml, "ref_idx", ref_idx);
            *ret = create_dielectric( ref_idx );
        }break;
        default:
        {
            fprintf(stderr, "TOML error = failed to get material_type: type = %s?\n", mat_type_str);
            return false;
        }
    }
    
    return success;
}
bool load_object( toml_table_t* object_toml, shape* ret)
{
    material mat;
    if( !load_material( object_toml, &mat) )
        return false;
   
    toml_table_t* shape_toml = toml_table_in( object_toml, "shape" );
    if( !shape_toml )
    {
        fprintf(stderr, "TOML error = failed to read 'shape' in object\n");
        return false;
    }
    
    const char* shape_type_str = toml_raw_in( shape_toml, "type" );
    if( !shape_type_str )
    {
        fprintf(stderr, "TOML error = failed to shape read type\n");
        return false;    
    }
        
    bool success = true;
    shape_type type = string_to_shape_type( shape_type_str );
    switch( type )
    {
        case ST_SPHERE:
        {
            FVec3 center;
            float radius;
            success &= my_toml_table_to_vector3( shape_toml, "center", center);
            success &= my_toml_raw_to_float( shape_toml, "radius", radius);
            
            if( success )
            {
                *ret = create_sphere( mat, center, radius );
            }
        }break;
        default:
        {
            fprintf(stderr, "TOML error = failed to get shape_type: type = %s?\n", shape_type_str);
            return false;
        }
    }
    
    return success;
}

bool load_objects( toml_table_t* parent_toml, shape*& shapes, int& num_objs)
{
    toml_array_t* objects_toml = toml_array_in( parent_toml, "object");
    if( !objects_toml )
    {
        fprintf(stderr, "TOML error = failed to read array of objects\n");
        return false;
    }
    
    num_objs = toml_array_nelem( objects_toml );
    shapes = (shape*)malloc( sizeof(shape) *num_objs );
    for(int i = 0; i < num_objs; i++)
    {
        toml_table_t* curr = toml_table_at( objects_toml, i );
        if( !load_object( curr, shapes + i) )
        {
            fprintf(stderr, "TOML error = failed to read object %i\n", i);
            free(shapes);
            num_objs = 0;
            shapes = nullptr;
            return false;
        }
    }
    return true;
}

bool load_camera( toml_table_t* parent_toml, camera& ret )
{
    toml_table_t* camera_table = toml_table_in( parent_toml, "camera");
    if( !camera_table )
    {
        fprintf(stderr, "TOML error = failed to load camera\n");
        return false;
    }
       
    bool success = true;
    
    FVec3 pos;
    FVec3 lookAt;
    float aperture;
    float vfov_degree;
    float aspect_ratio;
    
    // must success to read these values
    success &= my_toml_table_to_vector3( camera_table, "pos", pos);
    success &= my_toml_table_to_vector3( camera_table, "lookAt", lookAt);
    
    success &= my_toml_raw_to_float( camera_table, "aspect_ratio", aspect_ratio);
    success &= my_toml_raw_to_float( camera_table, "vfov_degree", vfov_degree );
    success &= my_toml_raw_to_float( camera_table, "aperture", aperture );
    
    if( !success )
    {
        return false;
    }
    
    // optional values to read and here are their default values 
    FVec3 vup = {0, 1, 0};
    my_toml_table_to_vector3( camera_table, "vup", vup );
    float focus_dist = fvec3_length(pos-lookAt); 
    my_toml_raw_to_float( camera_table, "focus_dist", focus_dist );
   
    
    ret = create_camera( 
    pos, lookAt, vup, aspect_ratio, vfov_degree, focus_dist, aperture);
    
    return true;
}

bool load_app( const char * const scene, app_data& app )
{
    FILE* file = fopen( scene, "r" );
    const int err_buff_size = 255;
    char err_buffer[err_buff_size + 1] = {0};
    
    if( !file  )
    {
        strncpy( err_buffer, scene, err_buff_size );
        err_buffer[err_buff_size] = 0;
        fprintf(stderr, "Load error = failed to open file %s", err_buffer);
        return false;
    }
    
    toml_table_t* doc = toml_parse_file(file, err_buffer, err_buff_size);
    fclose(file);
    
    if( !doc )
    {
        err_buffer[err_buff_size] = 0;
        fprintf(stderr, "TOML error = %s", err_buffer);
        return false;
    }
    
    bool success = true;
    success &= my_toml_raw_to_int( doc, "nx", app.nx );
    success &= my_toml_raw_to_int( doc, "ny", app.ny );
    success &= my_toml_raw_to_int( doc, "ns", app.ns );
    success &= my_toml_raw_to_int( doc, "thread_x", app.thread_x );
    success &= my_toml_raw_to_int( doc, "thread_y", app.thread_y );
    success &= my_toml_raw_to_int( doc, "max_reflect", app.max_reflect);
    
    if( !success || 
        !load_camera( doc, app.camera ) || 
        !load_objects( doc, app.shapes, app.num_objs) )
    {
        toml_free( doc );
        return false;
    }
    
    toml_free( doc );
    return true;
}


void default_scene( app_data& app )
{
    app.nx = 500;
    app.ny = 500;
    app.ns = 10;
    app.thread_x = 8;
    app.thread_y = 8;
    app.max_reflect = 25;
    
    app.num_objs = 2;
    app.shapes = (shape*)malloc( sizeof(shape) *app.num_objs );
    for(int i = 0; i < app.num_objs; i++)
    {
        material mat = create_metal( {0.4f, 0.5f, 0.8f}, 0.5f );
        app.shapes[i] = create_sphere(mat, { i * 1.f, 0, i * 1.f }, 0.5f);
    }
    
    FVec3 pos = {-2,+2,1};
    FVec3 lookAt = {0,0,-1};
    
    app.camera = create_camera( pos, lookAt, {0,1,0}, 
    float(app.nx)/app.ny , 90, fvec3_length(pos-lookAt), 2);
}


int main(int argc, char **argv) {
    
    app_data app;
    if ( !load_app( "app_data.toml", app ) )
    {
        fprintf(stderr, "failed to load app data!");
        default_scene( app );
    }
 
    // Allocate framebuffer in the unified memory
    int num_pixels = app.nx * app.ny;
    unsigned char* fb;
    size_t fb_size = 3 * num_pixels * sizeof(unsigned char); 
    checkCudaError( cudaMallocManaged(&fb, fb_size) );
    
    // Allocate random number state(?) on the device global memory
    curandState *dev_rand_state;
    checkCudaError( cudaMalloc(&dev_rand_state, num_pixels*sizeof(curandState)) );
    
    
    // Allocate variables that are passed to and used in the device
    shape* dev_world;
    camera* dev_cam;
    unsigned int memsize = sizeof(shape) * app.num_objs;
    checkCudaError( cudaMalloc( &dev_world, memsize ) );
    checkCudaError( cudaMalloc( &dev_cam, sizeof(camera)) );

    // Copy host world and camera to device
    checkCudaError( cudaMemcpy(dev_world, app.shapes, memsize, cudaMemcpyHostToDevice) );
    checkCudaError( cudaMemcpy(dev_cam, &app.camera, sizeof(camera), cudaMemcpyHostToDevice) );
    
    // Determine the number of threads and blocks
    int block_x = (app.nx + app.thread_x - 1)/app.thread_x;
    int block_y = (app.ny + app.thread_y - 1)/app.thread_y;
    
    dim3 blocks(block_x, block_y);
    dim3 threads(app.thread_x, app.thread_y);
    
    // Initialize random state
    rand_init<<<blocks, threads>>>(dev_rand_state, app.nx, app.ny);
    checkCudaError( cudaGetLastError() );
    checkCudaError( cudaDeviceSynchronize() );
    
    // Render objs in the framebuffer
    render<<<blocks, threads>>>( fb, app.nx, app.ny, app.ns,
                                 dev_cam, dev_world, app.num_objs, app.max_reflect,
                                 dev_rand_state);
    checkCudaError( cudaGetLastError() );
    checkCudaError( cudaDeviceSynchronize() );
  
  
      
    // Deallocate heap world variables
    checkCudaError( cudaFree(dev_world) );
    checkCudaError( cudaFree(dev_cam) );
    
    // Deallocate random states
    checkCudaError( cudaFree(dev_rand_state) );
    
    print_ppm(fb, app.nx, app.ny);
    
    // Deallocate framebuffer
    checkCudaError( cudaFree(fb) );

    return 0;  
}
