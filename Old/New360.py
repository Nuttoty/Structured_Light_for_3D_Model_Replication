import open3d as o3d
import numpy as np
import copy

def preprocess_point_cloud(pcd, voxel_size):
    print(" :: Downsampling with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(" :: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(" :: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(" :: RANSAC alignment on feature histograms...")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def main():
    # 1. ADJUST THIS SCALE
    # If your model is in mm, use e.g., 5.0. If in meters, use 0.05
    voxel_size = 0.005 

    print("1. Load two point clouds and show initial state")
    source = o3d.io.read_point_cloud("scan_0.ply") # The 30 degree scan
    target = o3d.io.read_point_cloud("scan_1.ply") # The 0 degree scan (The 'Anchor')

    # VISUALIZATION: Show them before alignment (likely far apart)
    source.paint_uniform_color([1, 0.706, 0]) # Yellow
    target.paint_uniform_color([0, 0.651, 0.929]) # Blue
    # o3d.visualization.draw_geometries([source, target], window_name="Before Alignment")

    # 2. Extract Geometric Features (FPFH)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # 3. Run Global Registration (RANSAC)
    # This finds the alignment WITHOUT knowing the center of rotation
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    
    # 4. Refine with ICP (Tighten the fit)
    print(" :: Point-to-plane ICP refinement")
    result_icp = o3d.pipelines.registration.registration_icp(
        source_down, target_down, voxel_size, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    # 5. Apply the found transformation to the original source cloud
    print(" :: Applying transformation")
    source.transform(result_icp.transformation)

    # 6. Visualize Result
    o3d.visualization.draw_geometries([source, target], 
                                      window_name="Automatic Alignment Result")
    
    print("\n--- FOUND TRANSFORMATION MATRIX ---")
    print(result_icp.transformation)
    print("-----------------------------------")
    print("You can now apply this matrix to your other scans relative to each other.")

if __name__ == "__main__":
    main()