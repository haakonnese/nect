#include <torch/torch.h>
#include <ATen/ATen.h>
#include <cmath>
#include <pybind11/pybind11.h>
#include <curand_kernel.h>

struct BaseGeometry {
    std::tuple<int, int> nDetector;
    std::tuple<float, float> dDetector;
    std::tuple<float, float, float> offOrigin;
    std::tuple<float, float> offDetector;
    std::tuple<float, float, float> rotDetector;
    float COR;
    BaseGeometry(std::tuple<int, int> nDetector,
                 std::tuple<float, float> dDetector,
                 std::tuple<float, float, float> offOrigin,
                 std::tuple<float, float> offDetector,
                 std::tuple<float, float, float> rotDetector,
                 float COR) : nDetector(nDetector),
                               dDetector(dDetector),
                               offOrigin(offOrigin),
                               offDetector(offDetector),
                               rotDetector(rotDetector),
                               COR(COR) {};
};
struct BaseConeGeometry: public BaseGeometry {
    float distance_source_to_origin;
    float distance_origin_to_detector;
    BaseConeGeometry(float distance_source_to_origin,
                     float distance_origin_to_detector,
                     std::tuple<int, int> nDetector,
                     std::tuple<float, float> dDetector,
                     std::tuple<float, float, float> offOrigin,
                     std::tuple<float, float> offDetector,
                     std::tuple<float, float, float> rotDetector,
                     float COR) : BaseGeometry(nDetector, dDetector, offOrigin, offDetector, rotDetector, COR),
                                   distance_source_to_origin(distance_source_to_origin),
                                   distance_origin_to_detector(distance_origin_to_detector) {};
};

struct BaseGeometryCylindrical {
    float object_radius;
    float object_height;
    float remove_factor_top;
    float remove_factor_bottom;
    BaseGeometryCylindrical(float object_radius,
                            float object_height,
                            float remove_factor_top,
                            float remove_factor_bottom) : object_radius(object_radius),
                                                          object_height(object_height),
                                                          remove_factor_top(remove_factor_top),
                                                          remove_factor_bottom(remove_factor_bottom) {};
};
struct ParallelGeometryCylindrical : public BaseGeometryCylindrical, public BaseGeometry {
    ParallelGeometryCylindrical(std::tuple<int, int> nDetector,
                                std::tuple<float, float> dDetector,
                                float object_radius,
                                float object_height,
                                std::tuple<float, float, float> offOrigin,
                                std::tuple<float, float> offDetector,
                                std::tuple<float, float, float> rotDetector,
                                float COR,
                                float remove_factor_top,
                                float remove_factor_bottom) : BaseGeometry(nDetector, dDetector, offOrigin, offDetector, rotDetector, COR),
                                                              BaseGeometryCylindrical( object_radius, object_height, remove_factor_top, remove_factor_bottom) {};
};

struct BaseGeometryVoxel {
    std::tuple<float, float, float> sVoxel;
    std::tuple<float, float, float> dVoxel;
    BaseGeometryVoxel(std::tuple<float, float, float> sVoxel, std::tuple<float, float, float> dVoxel) : sVoxel(sVoxel), dVoxel(dVoxel) {};
};
struct ConeGeometryVoxel : public BaseConeGeometry, public BaseGeometryVoxel{
    ConeGeometryVoxel(float distance_source_to_origin,
                      float distance_origin_to_detector,
                      std::tuple<int, int> nDetector,
                      std::tuple<float, float> dDetector,
                      std::tuple<float, float, float> offOrigin,
                      std::tuple<float, float> offDetector,
                      std::tuple<float, float, float> rotDetector,
                      float COR,
                      std::tuple<float, float, float> sVoxel,
                      std::tuple<float, float, float> dVoxel) : 
                      BaseConeGeometry(distance_source_to_origin, distance_origin_to_detector, nDetector, dDetector, offOrigin, offDetector, rotDetector, COR),
                      BaseGeometryVoxel(sVoxel, dVoxel) {};
};

struct ParallelGeometryVoxel : public BaseGeometry, public BaseGeometryVoxel{
    ParallelGeometryVoxel(std::tuple<int, int> nDetector,
                          std::tuple<float, float> dDetector,
                          std::tuple<float, float, float> offOrigin,
                          std::tuple<float, float> offDetector,
                          std::tuple<float, float, float> rotDetector,
                          float COR,
                          std::tuple<float, float, float> sVoxel,
                          std::tuple<float, float, float> dVoxel) : BaseGeometry(nDetector, dDetector, offOrigin, offDetector, rotDetector, COR),
                                     BaseGeometryVoxel(sVoxel, dVoxel) {};
};

struct ConeGeometryCylindrical : public BaseConeGeometry, public BaseGeometryCylindrical { 
    ConeGeometryCylindrical(float distance_source_to_origin,
                            float distance_origin_to_detector,
                            std::tuple<int, int> nDetector,
                            std::tuple<float, float> dDetector,
                            std::tuple<float, float, float> offOrigin,
                            std::tuple<float, float> offDetector,
                            std::tuple<float, float, float> rotDetector,
                            float COR,
                            float object_radius,
                            float object_height,
                            float remove_factor_top,
                            float remove_factor_bottom) 
        : BaseConeGeometry(distance_source_to_origin, distance_origin_to_detector, nDetector, dDetector, offOrigin, offDetector, rotDetector, COR),
          BaseGeometryCylindrical(object_radius, object_height, remove_factor_top, remove_factor_bottom) {};
};

struct Point3D {
    double x, y, z;
};

__device__ void rotate_points(Point3D& start, Point3D& end, float angle_rad, const BaseGeometry& geometry) {
    float cos_angle = cos(angle_rad);
    float sin_angle = sin(angle_rad);
    start.y = start.y - geometry.COR;
    end.y = end.y - geometry.COR;
    start = {cos_angle * start.x - sin_angle * start.y, sin_angle * start.x + cos_angle * start.y, start.z};
    end = {cos_angle * end.x - sin_angle * end.y, sin_angle * end.x + cos_angle * end.y, end.z};
}

__device__ bool get_tid_ray_index(torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> random_ray_index,
                                  const BaseGeometry& geometry,
                                  int num_rays,
                                  int starting_ray_index,
                                  int& tid, 
                                  int& ray_index) {
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rays) {
        return false;
    }
    if (tid + starting_ray_index >= random_ray_index.size(0)) {
        return false;
    }
    ray_index = random_ray_index[tid+starting_ray_index];
    if (ray_index >= std::get<0>(geometry.nDetector) * std::get<1>(geometry.nDetector)) {
        return false;
    }
    return true;
}

__device__ bool isBetween(double value, double min, double max) {
    return (value >= min && value <= max);
}

__device__ void swap(double& a, double& b) {
    double temp = a;
    a = b;
    b = temp;
}

__device__ bool findIntersection(Point3D start, Point3D end, Point3D prismMin, Point3D prismMax, Point3D& intersectionPoint1, Point3D& intersectionPoint2) {
    // Check intersection along the x-axis
    double tMinX = (prismMin.x - start.x) / (end.x - start.x);
    double tMaxX = (prismMax.x - start.x) / (end.x - start.x);

    if (tMinX > tMaxX) {
        swap(tMinX, tMaxX);
    }
    // Check intersection along the y-axis
    double tMinY = (prismMin.y - start.y) / (end.y - start.y);
    double tMaxY = (prismMax.y - start.y) / (end.y - start.y);

    if (tMinY > tMaxY) swap(tMinY, tMaxY);

    // Check intersection along the z-axis
    double tMinZ = (prismMin.z - start.z) / (end.z - start.z);
    double tMaxZ = (prismMax.z - start.z) / (end.z - start.z);

    if (tMinZ > tMaxZ) swap(tMinZ, tMaxZ);

    double tMin = std::max(std::max(tMinX, tMinY), tMinZ);
    double tMax = std::min(std::min(tMaxX, tMaxY), tMaxZ);

    // Check if the line intersects with the prism
    if (tMin > tMax || !isBetween(tMin, 0.0, 1.0) || !isBetween(tMax, 0.0, 1.0)) {
        return false;
    }

    // Calculate the intersection point
    intersectionPoint1.x = start.x + tMin * (end.x - start.x);
    intersectionPoint1.y = start.y + tMin * (end.y - start.y);
    intersectionPoint1.z = start.z + tMin * (end.z - start.z);

    intersectionPoint2.x = start.x + tMax * (end.x - start.x);
    intersectionPoint2.y = start.y + tMax * (end.y - start.y);
    intersectionPoint2.z = start.z + tMax * (end.z - start.z);

    return true;
}

__device__ void rotate_detector(const BaseGeometry& geometry,
                                Point3D& p2) {
    float alpha_ = std::get<2>(geometry.rotDetector); // yaw
    float beta_ = std::get<1>(geometry.rotDetector); //  pitch
    float gamma_ = std::get<0>(geometry.rotDetector); // roll
    float c_alpha = cos(-alpha_);
    float c_beta = cos(-beta_);
    float c_gamma = cos(-gamma_);
    float s_alpha = sin(-alpha_);
    float s_beta = sin(-beta_);
    float s_gamma = sin(-gamma_);
    float R12 = c_alpha * s_beta * s_gamma - s_alpha * c_gamma;
    float R13 = c_alpha * s_beta * c_gamma + s_alpha * s_gamma;
    float R22 = s_alpha * s_beta * s_gamma + c_alpha * c_gamma;
    float R23 = s_alpha * s_beta * c_gamma - c_alpha * s_gamma;
    float R32 = c_beta * s_gamma;
    float R33 = c_beta * c_gamma;
    float x = R12 * p2.y + R13 * p2.z;
    float y = R22 * p2.y + R23 * p2.z;
    float z = R32 * p2.y + R33 * p2.z;
    // printf("%f, %f : %f, %f : %f, %f \n", p2.x, x, p2.y, y, p2.z, z);
    p2.x = x;
    p2.y = y;
    p2.z = z;
}

__device__ void calc_xyz_detector_pos(const BaseGeometry& geometry,
                                      torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> random_offset,
                                      int ray_index,
                                      int tid,
                                      Point3D& p2, 
                                      float mag) {
    int detector_pixel_x = ray_index % std::get<1>(geometry.nDetector);
    int detector_pixel_z = ray_index / std::get<1>(geometry.nDetector);
    float detector_xy = (detector_pixel_x - (static_cast<float>(std::get<1>(geometry.nDetector)-1) / 2) + random_offset[tid][0]) * std::get<1>(geometry.dDetector) + std::get<1>(geometry.offDetector); //* static_cast<float>(std::get<1>(geometry.nDetector)) / (static_cast<float>(std::get<1>(geometry.nDetector))-1); 
    // detector_xy = detector_xy + geometry.COR * mag;
    p2.y = - detector_xy;
    p2.z = (detector_pixel_z - (static_cast<float>(std::get<0>(geometry.nDetector)-1) / 2) + random_offset[tid][1]) * std::get<0>(geometry.dDetector) + std::get<0>(geometry.offDetector); //* static_cast<float>(std::get<0>(geometry.nDetector)) / (static_cast<float>(std::get<0>(geometry.nDetector)) -1);
    p2.x = 0;
    // if (ray_index == 0 || ray_index == std::get<0>(geometry.nDetector)*std::get<1>(geometry.nDetector) -1) {
    //     printf("%f %f\n", p2.y, p2.z);
    // }
    if (std::get<0>(geometry.rotDetector) != 0 || std::get<1>(geometry.rotDetector) != 0 || std::get<2>(geometry.rotDetector) != 0) {
        rotate_detector(geometry, p2);
    }
}

__device__ void calc_xyz_detector_pos_cone(const BaseConeGeometry& geometry,
                                           torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> random_offset,
                                           int ray_index,
                                           int tid,
                                           Point3D& p2) {
    float mag = (geometry.distance_origin_to_detector + geometry.distance_source_to_origin) / geometry.distance_source_to_origin;
    calc_xyz_detector_pos(geometry, random_offset, ray_index, tid, p2, mag);
    p2.x = p2.x + geometry.distance_origin_to_detector;

}

template <typename T>
__device__ bool start_end_helper(Point3D p1,
                                 Point3D direction,
                                 const T& geometry,
                                 float angle_rad,
                                 Point3D& start,
                                 Point3D& end) {
    float a = direction.x * direction.x + direction.y * direction.y;
    float b = 2 * p1.x * direction.x + 2 * (p1.y - geometry.COR) * direction.y;
    float c = p1.x * p1.x + p1.y * p1.y - geometry.object_radius * geometry.object_radius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant <= 0) {
        return false;
    } 
    float sqrt_discriminant = sqrt(discriminant);
    float two_a = 2 * a;
    float t1 = (-b - sqrt_discriminant) / (two_a);
    float t2 = (-b + sqrt_discriminant) / (two_a);
    start.x = (p1.x + t1 * direction.x); // / (2 * geometry.object_radius);
    start.y = (p1.y + t1 * direction.y); // / (2 * geometry.object_radius);
    start.z = (t1 * direction.z); // / (geometry.object_height);
    end.x = (p1.x + t2 * direction.x); // / (2 * geometry.object_radius);
    end.y = (p1.y + t2 * direction.y); // / (2 * geometry.object_radius);
    end.z = (t2 * direction.z); // / (geometry.object_height);
    // end.y = p1.y + t2 * direction.y;
    // end.z = t2 * direction.z;
    
    rotate_points(start, end, angle_rad, geometry);

    return true;
}

__device__ bool calc_start_end_points(const ParallelGeometryCylindrical& geometry,
                                      torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> random_offset,
                                      int ray_index,
                                      int tid,
                                      float angle_rad,
                                      Point3D& start,
                                      Point3D& end) {
    Point3D p2;
    // printf("Wrong\n");
    calc_xyz_detector_pos(geometry, random_offset, ray_index, tid, p2, 1);
    p2.x = p2.x + geometry.object_radius;
    Point3D p1 = {-p2.x, p2.y, p2.z};
    Point3D direction = {p2.x - p1.x, 0, 0};
    return start_end_helper(p1, direction, geometry, angle_rad, start, end);
}

__device__ bool calc_start_end_points(const ConeGeometryCylindrical& geometry,
                                      torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> random_offset,
                                      int ray_index,
                                      int tid,
                                      float angle_rad,
                                      Point3D& start,
                                      Point3D& end) {
    Point3D p1 = {-geometry.distance_source_to_origin, 0, 0};
    Point3D p2;
    calc_xyz_detector_pos_cone(geometry, random_offset, ray_index, tid, p2);
    
    Point3D direction = {p2.x - p1.x, p2.y, p2.z};
    // parametric line intersection with circle
    return start_end_helper(p1, direction, geometry, angle_rad, start, end);
    
}


__device__ bool calc_start_end_points(const ConeGeometryVoxel& geometry,
                                      torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> random_offset,
                                      int ray_index,
                                      int tid,
                                      float angle_rad,
                                      Point3D& start,
                                      Point3D& end) {
    Point3D p1 = {-geometry.distance_source_to_origin, 0, 0};
    Point3D p2;
    // printf("ray_index: %d\n", ray_index);
    calc_xyz_detector_pos_cone(geometry, random_offset, ray_index, tid, p2);
    // printf("p2: %f, %f, %f\n", p2.x, p2.y, p2.z);
    rotate_points(p1, p2, angle_rad, geometry);
    // printf("p1 rot: %f, %f, %f\n", p1.x, p1.y, p1.z);
    // printf("p2 rot: %f, %f, %f\n", p2.x, p2.y, p2.z);
    float sX = (std::get<2>(geometry.sVoxel) - std::get<2>(geometry.dVoxel)/2) / 2;
    float sY = (std::get<1>(geometry.sVoxel) - std::get<1>(geometry.dVoxel)/2) / 2;
    float sZ = (std::get<0>(geometry.sVoxel) - std::get<0>(geometry.dVoxel)/2) / 2;
    Point3D min = {-sX, -sY, -sZ};
    Point3D max = {sX, sY, sZ};
    if(!findIntersection(p1, p2, min, max, start, end)){
        return false;
    }
    float eps = 1e-5;
    if (start.z -eps > std::get<0>(geometry.sVoxel)/2 || start.z + eps < -std::get<0>(geometry.sVoxel)/2  || start.x -eps> std::get<2>(geometry.sVoxel)/2 || start.x + eps < -std::get<2>(geometry.sVoxel)/2 || start.y -eps> std::get<1>(geometry.sVoxel)/2 || start.y + eps < -std::get<1>(geometry.sVoxel)/2) {
        printf("start: %f, %f, %f\n", start.x, start.y, start.z);
    }
    if (end.z -eps> std::get<0>(geometry.sVoxel)/2 || end.z + eps < -std::get<0>(geometry.sVoxel)/2 || end.x -eps> std::get<2>(geometry.sVoxel)/2 || end.x + eps < -std::get<2>(geometry.sVoxel)/2 || end.y -eps> std::get<1>(geometry.sVoxel)/2 || end.y + eps < -std::get<1>(geometry.sVoxel)/2) {
        printf("end: %f, %f, %f, %f, %f, %f\n", end.x, end.y, end.z, std::get<2>(geometry.sVoxel), std::get<1>(geometry.sVoxel), std::get<0>(geometry.sVoxel));
    }
    
    // printf("start: %f, %f, %f\n", start.x, start.y, start.z);
    return true;
}

__device__ bool calc_start_end_points(const ParallelGeometryVoxel& geometry,
                                      torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> random_offset,
                                      int ray_index,
                                      int tid,
                                      float angle_rad,
                                      Point3D& start,
                                      Point3D& end) {
    
    Point3D p2;
    // printf("ray_index: %d\n", ray_index);
    calc_xyz_detector_pos(geometry, random_offset, ray_index, tid, p2, 1);
    p2.x = p2.x + sqrt(std::get<2>(geometry.sVoxel) * std::get<2>(geometry.sVoxel) + std::get<1>(geometry.sVoxel) * std::get<1>(geometry.sVoxel));
    Point3D p1 = {-p2.x, p2.y, p2.z};
    // printf("p2: %f, %f, %f\n", p2.x, p2.y, p2.z);
    rotate_points(p1, p2, angle_rad, geometry);
    // printf("p1 rot: %f, %f, %f\n", p1.x, p1.y, p1.z);
    // printf("p2 rot: %f, %f, %f\n", p2.x, p2.y, p2.z);
    // float sX = (std::get<2>(geometry.sVoxel) - std::get<2>(geometry.dVoxel)) / 2;
    // float sY = (std::get<1>(geometry.sVoxel) - std::get<1>(geometry.dVoxel)) / 2;
    // float sZ = (std::get<0>(geometry.sVoxel) - std::get<0>(geometry.dVoxel)) / 2;
    float sX = (std::get<2>(geometry.sVoxel) - std::get<2>(geometry.dVoxel)/2) / 2;
    float sY = (std::get<1>(geometry.sVoxel) - std::get<1>(geometry.dVoxel)/2) / 2;
    float sZ = (std::get<0>(geometry.sVoxel) - std::get<0>(geometry.dVoxel)/2) / 2;
    Point3D min = {-sX, -sY, -sZ};
    Point3D max = {sX, sY, sZ};
    if(!findIntersection(p1, p2, min, max, start, end)){
        return false;
    }
    // printf("start: %f, %f, %f\n", start.x, start.y, start.z);
    return true;
}


__device__ void calc_all_points(Point3D start, 
                                Point3D end, 
                                torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> ray_points,
                                torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> random_numbers,
                                torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> distances,
                                int tid,
                                int num_points_per_ray,
                                float max_ray_distance_per_point,
                                bool uniform_ray_spacing,
                                const BaseGeometryCylindrical& geometry) {
    Point3D step = {end.x - start.x, end.y - start.y, end.z - start.z};
    float ray_distance = sqrt(step.x * step.x + step.y * step.y + step.z * step.z);
    
    if (uniform_ray_spacing) {
        distances[tid] = max_ray_distance_per_point;
       //          direction vector        total length of vector                 scaled by 
        step.x = (step.x / ray_distance) * max_ray_distance_per_point / (2 * geometry.object_radius);
        step.y = (step.y / ray_distance) * max_ray_distance_per_point / (2 * geometry.object_radius);
        step.z = (step.z / ray_distance) * max_ray_distance_per_point / (geometry.object_height);
    }
    else {
        distances[tid] = ray_distance / num_points_per_ray;
        step.x = (step.x / num_points_per_ray) / (2 * geometry.object_radius);
        step.y = (step.y / num_points_per_ray) / (2 * geometry.object_radius);
        step.z = (step.z / num_points_per_ray) / (geometry.object_height);
    }
    
    start.x /= 2 * geometry.object_radius;
    end.x /= 2 * geometry.object_radius;
    start.y /= 2 * geometry.object_radius;
    end.y /= 2 * geometry.object_radius;
    start.z /= (geometry.object_height);
    
    // random offset to avoid aliasing. The offset is between -0.5 and 0.5. 
    float random_offset_start = random_numbers[tid];

    start = {start.x + 0.5 + random_offset_start * step.x, 
             start.y + 0.5 + random_offset_start * step.y, 
             start.z + 0.5 + random_offset_start * step.z};
    end = {end.x + 0.5, end.y + 0.5, end.z};
    start.z = min(max(start.z, 0.0f), 1.0f);
    start.y = min(max(start.y, 0.0f), 1.0f);
    start.x = min(max(start.x, 0.0f), 1.0f);
    end.z = min(max(end.z, 0.0f), 1.0f);
    end.y = min(max(end.y, 0.0f), 1.0f);
    end.x = min(max(end.x, 0.0f), 1.0f);
            
    // calculate the x, y, and z coordinates of each point along the ray
    # pragma omp simd
    for (int i = 0; i < num_points_per_ray; i++) {
        float pos_x = start.x + i * step.x;
        float pos_y = start.y + i * step.y;
        float pos_z = start.z + i * step.z;
        if (step.x > 0) {
            if (pos_x > end.x){
                break;
            }
        }
        else if (step.x < 0) {
            if (pos_x < end.x) {
                break;
            }
        }
        else if (step.y > 0) {
            if (pos_y > end.y) {
                break;
            }
        }
        else if (step.y < 0) {
            if (pos_y < end.y) {
                break;
            }
        }
        if (pos_z < 0 + geometry.remove_factor_top || pos_z > 1 - geometry.remove_factor_bottom) {
            break;
        }
        
        ray_points[tid][i][0] = pos_z;
        ray_points[tid][i][1] = pos_y;
        ray_points[tid][i][2] = pos_x;
    }
}

__device__ void calc_all_points(Point3D start, 
                                Point3D end, 
                                torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> ray_points,
                                torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> random_numbers,
                                torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> distances,
                                int tid,
                                int num_points_per_ray,
                                float max_ray_distance_per_point,
                                bool uniform_ray_spacing,
                                const BaseGeometryVoxel& geometry) {
    
    Point3D step = {end.x - start.x, end.y - start.y, end.z - start.z};
    float ray_distance = sqrt(step.x * step.x + step.y * step.y + step.z * step.z);
    if (uniform_ray_spacing) {
        distances[tid] = max_ray_distance_per_point;
        step.x = (step.x / ray_distance) * max_ray_distance_per_point / std::get<2>(geometry.sVoxel);
        step.y = (step.y / ray_distance) * max_ray_distance_per_point / std::get<1>(geometry.sVoxel);
        step.z = (step.z / ray_distance) * max_ray_distance_per_point / std::get<0>(geometry.sVoxel);
        
    }
    else {
        distances[tid] = ray_distance / num_points_per_ray;
        step.x = (step.x / num_points_per_ray) / std::get<2>(geometry.sVoxel);
        step.y = (step.y / num_points_per_ray) / std::get<1>(geometry.sVoxel);
        step.z = (step.z / num_points_per_ray) / std::get<0>(geometry.sVoxel);
    }
    start.x /= (std::get<2>(geometry.sVoxel));
    end.x /= (std::get<2>(geometry.sVoxel));
    start.y /= (std::get<1>(geometry.sVoxel));
    end.y /= (std::get<1>(geometry.sVoxel));
    start.z /= (std::get<0>(geometry.sVoxel));
    end.z /= (std::get<0>(geometry.sVoxel));
    
    // start.y /= std::get<1>(geometry.sVoxel);
    // end.y /= std::get<1>(geometry.sVoxel);
    // start.z /= std::get<0>(geometry.sVoxel);
    // end.z /= std::get<0>(geometry.sVoxel);
    // start = {start.x + std::get<2>(geometry.dVoxel) / std::get<2>(geometry.sVoxel),
    //          start.y + std::get<1>(geometry.dVoxel) / std::get<1>(geometry.sVoxel),
    //          start.z + std::get<0>(geometry.dVoxel) / std::get<0>(geometry.sVoxel)};
    // end = {end.x + std::get<2>(geometry.dVoxel) / std::get<2>(geometry.sVoxel), 
    //        end.y + std::get<1>(geometry.dVoxel) / std::get<1>(geometry.sVoxel),
    //        end.z + std::get<0>(geometry.dVoxel) / std::get<0>(geometry.sVoxel)};
    float random_offset_start = random_numbers[tid];
    float eps = 1e-5;

    start = {start.x + 0.5 + random_offset_start * step.x, 
             start.y + 0.5 + random_offset_start * step.y, 
             start.z + 0.5 + random_offset_start * step.z};
    end = {end.x + 0.5 + step.x, end.y + 0.5 + step.y, end.z + 0.5 + step.z};
    start.z = min(max(start.z, 0.0f), 1.0f);
    start.y = min(max(start.y, 0.0f), 1.0f);
    start.x = min(max(start.x, 0.0f), 1.0f);
    end.z = min(max(end.z, 0.0f), 1.0f);
    end.y = min(max(end.y, 0.0f), 1.0f);
    end.x = min(max(end.x, 0.0f), 1.0f);
    // if (start.z + eps < 0 || start.z - eps > 1 || start.y + eps < 0 || start.y - eps > 1 || start.x + eps < 0 || start.x - eps > 1) {
    //     printf("start: %f, %f, %f\n", start.x, start.y, start.z);
    // }
    // if (end.z + eps < 0 || end.z -eps > 1 || end.y + eps < 0 || end.y - eps > 1 || end.x + eps < 0 || end.x - eps > 1) {
    //     printf("end: %f, %f, %f\n", end.x, end.y, end.z);
    // }
    # pragma omp simd
    for (int i = 0; i < num_points_per_ray; i++) {
        float pos_x = start.x + i * step.x;
        float pos_y = start.y + i * step.y;
        float pos_z = start.z + i * step.z;
        if (step.x > 0) {
            if (pos_x - eps> end.x){
                break;
            }
        }
        else if (step.x < 0) {
            if (pos_x + eps < end.x) {
                break;
            }
        }
        if (step.y > 0) {
            if (pos_y - eps > end.y) {
                break;
            }
        }
        else if (step.y < 0) {
            if (pos_y + eps < end.y) {
                break;
            }
        }
        if (step.z > 0) {
            if (pos_z - eps > end.z) {
                break;
            }
        }
        else if (step.z < 0) {
            if (pos_z + eps < end.z) {
                break;
            }
        }
        ray_points[tid][i][0] = pos_z;
        ray_points[tid][i][1] = pos_y;
        ray_points[tid][i][2] = pos_x;
    }
}

template <typename T>
__global__ void sample_points(torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> ray_points,
                              torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> random_numbers,
                              torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> random_ray_index,
                              torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> random_offset,
                              torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> distances,
                              const T geometry,
                              float angle_rad, 
                              int num_points_per_ray,
                              int num_rays,
                              int starting_ray_index,
                              float max_ray_distance_per_point,
                              bool uniform_ray_spacing) {
    int tid;
    int ray_index;
    if (!get_tid_ray_index(random_ray_index, geometry, num_rays, starting_ray_index, tid, ray_index)) {
        return;
    }
    Point3D start;
    Point3D end;
    if(!calc_start_end_points(geometry, random_offset, ray_index, tid, angle_rad, start, end)){
        return;
    } 
    calc_all_points(start, end, ray_points, random_numbers, distances, tid, num_points_per_ray, max_ray_distance_per_point, uniform_ray_spacing, geometry);
}



template <typename T>
std::tuple<torch::Tensor, torch::Tensor> sample(const torch::Tensor& random_ray_index,
                     const T& geometry,
                     float angle_rad,
                     int num_points_per_ray,
                     int num_rays,
                     int starting_ray_index,
                     float max_ray_distance_per_point,
                     bool unform_ray_spacing,
                     float random_detector_offset,
                     int device) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA, device);
    torch::Tensor points = torch::zeros({num_rays, num_points_per_ray, 3}, options);
    torch::Tensor random_numbers = torch::rand({num_rays}, options) - 0.5;
    // torch::Tensor random_numbers = torch::zeros({num_rays}, options);
    // torch::Tensor random_offset = (torch::rand({num_rays, 2}, options) - 0.5) / 2;
    torch::Tensor random_offset = (torch::rand({num_rays, 2}, options) - 0.5) * random_detector_offset;
    torch::Tensor distances = torch::zeros({num_rays}, options);
    int threads_per_block = 256;
    int blocks_per_grid = (num_rays + threads_per_block - 1) / threads_per_block;
    // printf("Type geometry: %s\n", typeid(geometry).name());
    sample_points<<<blocks_per_grid, threads_per_block>>>(points.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                                                          random_numbers.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                          random_ray_index.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                                          random_offset.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                                          distances.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                          geometry,
                                                          angle_rad,
                                                          num_points_per_ray,
                                                          num_rays, 
                                                          starting_ray_index,
                                                          max_ray_distance_per_point,
                                                          unform_ray_spacing);
    // Check for errors
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
    
    return {points, distances};

}   

void bindParallelGeometryCylindrical(pybind11::module &m) {
    pybind11::class_<ParallelGeometryCylindrical>(m, "ParallelGeometryCylindrical")
        .def(pybind11::init<std::tuple<int, int>, std::tuple<float, float>, float, float, std::tuple<float, float, float>, std::tuple<float, float>, std::tuple<float, float, float>, float, float, float>(),
             pybind11::arg("nDetector"),
             pybind11::arg("dDetector"),
             pybind11::arg("object_radius"),
             pybind11::arg("object_height"),
             pybind11::arg("offOrigin") = std::make_tuple(0.0, 0.0, 0.0),
             pybind11::arg("offDetector") = std::make_tuple(0.0, 0.0),
             pybind11::arg("rotDetector") = std::make_tuple(0.0, 0.0, 0.0),
             pybind11::arg("COR") = 0.0,
             pybind11::arg("remove_factor_top") = 0.0,
             pybind11::arg("remove_factor_bottom") = 0.0)
        .def_readwrite("nDetector", &ParallelGeometryCylindrical::nDetector)
        .def_readwrite("dDetector", &ParallelGeometryCylindrical::dDetector)
        .def_readwrite("offOrigin", &ParallelGeometryCylindrical::offOrigin)
        .def_readwrite("offDetector", &ParallelGeometryCylindrical::offDetector)
        .def_readwrite("rotDetector", &ConeGeometryVoxel::rotDetector)
        .def_readwrite("COR", &ParallelGeometryCylindrical::COR)
        .def_readwrite("object_radius", &ParallelGeometryCylindrical::object_radius)
        .def_readwrite("object_height", &ParallelGeometryCylindrical::object_height)
        .def_readwrite("remove_factor_top", &ParallelGeometryCylindrical::remove_factor_top)
        .def_readwrite("remove_factor_bottom", &ParallelGeometryCylindrical::remove_factor_bottom);
}


void bindConeGeometryCylindrical(pybind11::module &m) {
    pybind11::class_<ConeGeometryCylindrical>(m, "ConeGeometryCylindrical")
        .def(pybind11::init<float, float, std::tuple<int, int>, std::tuple<float, float>, std::tuple<float, float, float>, std::tuple<float, float>, std::tuple<float, float, float>, float, float, float, float, float>(),
             pybind11::arg("distance_source_to_origin"),
             pybind11::arg("distance_origin_to_detector"),
             pybind11::arg("nDetector"),
             pybind11::arg("dDetector"),
             pybind11::arg("offOrigin") = std::make_tuple(0.0, 0.0, 0.0),
             pybind11::arg("offDetector") = std::make_tuple(0.0, 0.0),
             pybind11::arg("rotDetector") = std::make_tuple(0.0, 0.0, 0.0),
             pybind11::arg("COR") = 0.0,
             pybind11::arg("object_radius"),
             pybind11::arg("object_height"),
             pybind11::arg("remove_factor_top") = 0.0,
             pybind11::arg("remove_factor_bottom") = 0.0)
        .def_readwrite("distance_source_to_origin", &ConeGeometryCylindrical::distance_source_to_origin)
        .def_readwrite("distance_origin_to_detector", &ConeGeometryCylindrical::distance_origin_to_detector)
        .def_readwrite("nDetector", &ConeGeometryCylindrical::nDetector)
        .def_readwrite("dDetector", &ConeGeometryCylindrical::dDetector)
        .def_readwrite("offOrigin", &ConeGeometryCylindrical::offOrigin)
        .def_readwrite("offDetector", &ConeGeometryCylindrical::offDetector)
        .def_readwrite("rotDetector", &ConeGeometryVoxel::rotDetector)
        .def_readwrite("COR", &ConeGeometryCylindrical::COR)
        .def_readwrite("object_radius", &ConeGeometryCylindrical::object_radius)
        .def_readwrite("object_height", &ConeGeometryCylindrical::object_height)
        .def_readwrite("remove_factor_top", &ConeGeometryCylindrical::remove_factor_top)
        .def_readwrite("remove_factor_bottom", &ConeGeometryCylindrical::remove_factor_bottom);
}

void bindConeGeometryVoxel(pybind11::module &m) {
    pybind11::class_<ConeGeometryVoxel>(m, "ConeGeometryVoxel")
        .def(pybind11::init<float, float, std::tuple<int, int>, std::tuple<float, float>, std::tuple<float, float, float>, std::tuple<float, float>, std::tuple<float, float, float>, float, std::tuple<float, float, float>, std::tuple<float, float, float>>(),
             pybind11::arg("distance_source_to_origin"),
             pybind11::arg("distance_origin_to_detector"),
             pybind11::arg("nDetector"),
             pybind11::arg("dDetector"),
             pybind11::arg("offOrigin") = std::make_tuple(0.0, 0.0, 0.0),
             pybind11::arg("offDetector") = std::make_tuple(0.0, 0.0),
             pybind11::arg("rotDetector") = std::make_tuple(0.0, 0.0, 0.0),
             pybind11::arg("COR") = 0.0,
             pybind11::arg("sVoxel"),
             pybind11::arg("dVoxel"))
        .def_readwrite("distance_source_to_origin", &ConeGeometryVoxel::distance_source_to_origin)
        .def_readwrite("distance_origin_to_detector", &ConeGeometryVoxel::distance_origin_to_detector)
        .def_readwrite("nDetector", &ConeGeometryVoxel::nDetector)
        .def_readwrite("dDetector", &ConeGeometryVoxel::dDetector)
        .def_readwrite("offOrigin", &ConeGeometryVoxel::offOrigin)
        .def_readwrite("offDetector", &ConeGeometryVoxel::offDetector)
        .def_readwrite("rotDetector", &ConeGeometryVoxel::rotDetector)
        .def_readwrite("COR", &ConeGeometryVoxel::COR)
        .def_readwrite("sVoxel", &ParallelGeometryVoxel::sVoxel)
        .def_readwrite("dVoxel", &ParallelGeometryVoxel::dVoxel);
}

void bindParallelGeometryVoxel(pybind11::module &m) {
    pybind11::class_<ParallelGeometryVoxel>(m, "ParallelGeometryVoxel")
        .def(pybind11::init<std::tuple<int, int>, std::tuple<float, float>, std::tuple<float, float, float>, std::tuple<float, float>, std::tuple<float, float, float>, float, std::tuple<float, float, float>, std::tuple<float, float, float>>(),
             pybind11::arg("nDetector"),
             pybind11::arg("dDetector"),
             pybind11::arg("offOrigin") = std::make_tuple(0.0, 0.0, 0.0),
             pybind11::arg("offDetector") = std::make_tuple(0.0, 0.0),
             pybind11::arg("rotDetector") = std::make_tuple(0.0, 0.0, 0.0),
             pybind11::arg("COR") = 0.0,
             pybind11::arg("sVoxel"),
             pybind11::arg("dVoxel"))
        .def_readwrite("nDetector", &ParallelGeometryVoxel::nDetector)
        .def_readwrite("dDetector", &ParallelGeometryVoxel::dDetector)
        .def_readwrite("offOrigin", &ParallelGeometryVoxel::offOrigin)
        .def_readwrite("offDetector", &ParallelGeometryVoxel::offDetector)
        .def_readwrite("rotDetector", &ConeGeometryVoxel::rotDetector)
        .def_readwrite("COR", &ParallelGeometryVoxel::COR)
        .def_readwrite("sVoxel", &ParallelGeometryVoxel::sVoxel)
        .def_readwrite("dVoxel", &ParallelGeometryVoxel::dVoxel);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    bindConeGeometryCylindrical(m);
    bindConeGeometryVoxel(m);
    bindParallelGeometryCylindrical(m);
    bindParallelGeometryVoxel(m);
    m.def("sample",
          &sample<ConeGeometryCylindrical>,
          "CT scan (CUDA)",
          pybind11::arg("random_ray_index"),
          pybind11::arg("geometry"),
          pybind11::arg("angle_rad"),
          pybind11::arg("num_points_per_ray"),
          pybind11::arg("num_rays"),
          pybind11::arg("starting_ray_index"),
          pybind11::arg("max_ray_distance_per_point"),
          pybind11::arg("uniform_ray_spacing"),
          pybind11::arg("random_detector_offset"),
          pybind11::arg("device"));
    m.def("sample",
          &sample<ConeGeometryVoxel>,
          "CT scan (CUDA)",
          pybind11::arg("random_ray_index"),
          pybind11::arg("geometry"),
          pybind11::arg("angle_rad"),
          pybind11::arg("num_points_per_ray"),
          pybind11::arg("num_rays"),
          pybind11::arg("starting_ray_index"),
          pybind11::arg("max_ray_distance_per_point"),
          pybind11::arg("uniform_ray_spacing"),
          pybind11::arg("random_detector_offset"),
          pybind11::arg("device"));
    m.def("sample",
          &sample<ParallelGeometryCylindrical>,
          "CT scan (CUDA)",
          pybind11::arg("random_ray_index"),
          pybind11::arg("geometry"),
          pybind11::arg("angle_rad"),
          pybind11::arg("num_points_per_ray"),
          pybind11::arg("num_rays"),
          pybind11::arg("starting_ray_index"),
          pybind11::arg("max_ray_distance_per_point"),
          pybind11::arg("uniform_ray_spacing"),
          pybind11::arg("random_detector_offset"),
          pybind11::arg("device"));
    m.def("sample",
        &sample<ParallelGeometryVoxel>,
        "CT scan (CUDA)",
        pybind11::arg("random_ray_index"),
        pybind11::arg("geometry"),
        pybind11::arg("angle_rad"),
        pybind11::arg("num_points_per_ray"),
        pybind11::arg("num_rays"),
        pybind11::arg("starting_ray_index"),
        pybind11::arg("max_ray_distance_per_point"),
        pybind11::arg("uniform_ray_spacing"),
        pybind11::arg("random_detector_offset"),
        pybind11::arg("device"));
}