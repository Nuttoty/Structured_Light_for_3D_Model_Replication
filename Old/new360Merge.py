import open3d as o3d
import numpy as np
import copy
import os
import re

def get_sorted_file_paths(folder_path, extension=".ply"):
    file_paths = [
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) 
        if f.lower().endswith(extension)
    ]
    
    def extract_number(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0
        
    file_paths.sort(key=extract_number)
    return file_paths

def preprocess_point_cloud(pcd, voxel_size):
    """Downsamples, estimates normals, and extracts FPFH features for global matching."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        
    return pcd_down, pcd_fpfh

def load_point_clouds(folder_path, voxel_size):
    pcds = []
    pcds_down = []
    fpfhs = []
    file_paths = get_sorted_file_paths(folder_path, extension=".ply")
    
    if not file_paths:
        print(f"No .ply files found in {folder_path}")
        return pcds, pcds_down, fpfhs

    print(f"Found {len(file_paths)} scans. Extracting geometric features...")
    
    for path in file_paths:
        pcd = o3d.io.read_point_cloud(path)
        
        # ---> FIX: Estimate normals on the ORIGINAL point cloud before anything else <---
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        
        # Preprocess the downsampled version for global registration
        pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size)
        
        pcds.append(pcd)
        pcds_down.append(pcd_down)
        fpfhs.append(pcd_fpfh)
        
    return pcds, pcds_down, fpfhs

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """Uses RANSAC to globally align the 30-degree gaps based on geometry."""
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def pairwise_registration(source, target, source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    print("  -> Running Global Registration (RANSAC)...")
    ransac_result = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    
    print("  -> Running Fine ICP...")
    # Feed the RANSAC alignment into ICP so ICP knows exactly where to start
    distance_threshold = voxel_size * 0.4
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold,
        ransac_result.transformation, # <-- This is the magic bridge
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
        
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, distance_threshold,
        icp_fine.transformation)
        
    return transformation_icp, information_icp

def full_registration(pcds, pcds_down, fpfhs, voxel_size):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    
    n_pcds = len(pcds)
    
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            # Match sequential scans and the loop closure
            if target_id == source_id + 1 or (source_id == 0 and target_id == n_pcds - 1):
                print(f"\nAligning scan {source_id} to {target_id}...")
                
                transformation_icp, information_icp = pairwise_registration(
                    pcds[source_id], pcds[target_id], 
                    pcds_down[source_id], pcds_down[target_id],
                    fpfhs[source_id], fpfhs[target_id], voxel_size)
                
                if target_id == source_id + 1:
                    odometry = np.dot(transformation_icp, odometry)
                    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                    
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                    source_id, target_id, transformation_icp, information_icp, uncertain=False))
    
    print("\nRunning Pose Graph Optimization...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=voxel_size * 0.4,
        edge_prune_threshold=0.25,
        reference_node=0)
        
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)
    
    print("Transforming points...")
    for point_id in range(n_pcds):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        
    return pcds

def main():
    folder_path = input("Enter the folder path containing your 30-degree scans: ").strip()
    folder_path = folder_path.strip('\'"')
    
    if not os.path.isdir(folder_path):
        print("Error: Invalid folder path.")
        return

    # IMPORTANT: Voxel size dictates how the features are matched.
    # If your object is small (like a shoe), try a smaller number (e.g., 0.005).
    # If it's a room, try a larger number (e.g., 0.05).
    voxel_size = 1.0  
    
    # 1. Load and Preprocess Data (Extracting Features)
    pcds, pcds_down, fpfhs = load_point_clouds(folder_path, voxel_size)
    
    if not pcds:
        return

    # 2. Registration (Global RANSAC + Local ICP)
    aligned_pcds = full_registration(pcds, pcds_down, fpfhs, voxel_size)
    
    # 3. Merge & Cleanup
    print("\nMerging point clouds...")
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(aligned_pcds)):
        pcd_combined += aligned_pcds[point_id]

    print("Post-processing (Removing noise)...")
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size * 0.5)
    
    cl, ind = pcd_combined_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_final = pcd_combined_down.select_by_index(ind)
    pcd_final.estimate_normals()
    
    # ---------------------------------------------------------
    # ---> ADD THESE LINES TO SAVE THE MERGED POINT CLOUD <---
    # ---------------------------------------------------------
    print("Saving merged point cloud...")
    o3d.io.write_point_cloud(os.path.join(folder_path, "merged_pointcloud.ply"), pcd_final)
    print("Point cloud saved to 'merged_pointcloud.ply' in your folder.")
    # ---------------------------------------------------------
    
    # 4. Mesh Generation
    print("Generating Mesh (Poisson Surface Reconstruction)...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_final, depth=9)
    
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    o3d.visualization.draw_geometries([mesh], window_name="Final Merged Mesh")
    
    o3d.io.write_triangle_mesh("merged_result.stl", mesh)
    print("Saved to merged_result.stl")

if __name__ == "__main__":
    main()