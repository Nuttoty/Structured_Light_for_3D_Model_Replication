import open3d as o3d
import numpy as np
import copy

def load_point_clouds(voxel_size=0.0):
    pcds = []
    # UPDATE THIS: List your actual file paths here
    file_paths = ["scan_00.ply", "scan_30.ply", "scan_60.ply"] 
    
    for path in file_paths:
        pcd = o3d.io.read_point_cloud(path)
        # Pre-processing: Downsample and estimate normals (crucial for ICP)
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcds.append(pcd)
    return pcds

def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    print("Applying pairwise registration...")
    
    # 1. Coarse Registration (Point-to-Plane)
    # If your 30-degree scans are already roughly aligned by the scanner, you can skip to Fine.
    # If they are arbitrary, use RANSAC here. 
    # Assuming rough alignment exists or is close:
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    # 2. Fine Registration (Point-to-Plane)
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
        
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
        
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    
    n_pcds = len(pcds)
    
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            # We only match sequential scans (i and i+1) to build the chain
            # AND the loop closure (last scan to first scan)
            if target_id == source_id + 1 or (source_id == 0 and target_id == n_pcds - 1):
                
                print(f"Registering scan {source_id} to {target_id}...")
                transformation_icp, information_icp = pairwise_registration(
                    pcds[source_id], pcds[target_id], 
                    max_correspondence_distance_coarse, max_correspondence_distance_fine)
                
                if target_id == source_id + 1:
                    odometry = np.dot(transformation_icp, odometry)
                    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                    
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                    source_id, target_id, transformation_icp, information_icp, uncertain=False))
    
    print("Running Pose Graph Optimization...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
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
    voxel_size = 3 # Adjust based on your object scale (e.g. 0.02 meters)
    
    # 1. Load Data
    pcds = load_point_clouds(voxel_size)
    
    if not pcds:
        print("No point clouds loaded. Check file paths.")
        return

    # 2. Registration (Align)
    # Coarse threshold: usually 15x voxel size
    # Fine threshold: usually 1.5x voxel size
    aligned_pcds = full_registration(pcds, voxel_size * 15, voxel_size * 1.5)
    
    # 3. Merge & Cleanup
    print("Merging point clouds...")
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(aligned_pcds)):
        pcd_combined += aligned_pcds[point_id]

    print("Post-processing...")
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    
    # Remove outliers (noise reduction)
    cl, ind = pcd_combined_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_final = pcd_combined_down.select_by_index(ind)
    
    pcd_final.estimate_normals()
    
    # 4. Mesh Generation (Poisson)
    print("Generating Mesh (Poisson Surface Reconstruction)...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_final, depth=9)
    
    # Crop low density vertices (removes bubbles/artifacts outside the scan)
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    o3d.visualization.draw_geometries([mesh], window_name="Final Merged Mesh")
    
    # Save
    o3d.io.write_triangle_mesh("merged_result.stl", mesh)
    print("Saved to merged_result.stl")

if __name__ == "__main__":
    main()