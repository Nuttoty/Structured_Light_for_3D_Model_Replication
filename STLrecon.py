import numpy as np
import open3d as o3d
import sys
import os

# --- CONFIGURATION ---
# "watertight" = Fills holes, creates a closed solid (Poisson Reconstruction)
# "surface"    = Connects dots exactly. Leaves holes if data is missing (Ball Pivoting)
MODE = "watertight"  

# ---------------------

def create_mesh_watertight(pcd, depth=10):
    """
    Method 1: Poisson Reconstruction
    Creates a closed, watertight 'bubble'. Good for printing.
    """
    print(f"Mode: WATERTIGHT (Poisson Depth={depth})")
    # Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, linear_fit=False)
    
    # Trim very low density noise to reduce the 'paper' effect slightly
    densities = np.asarray(densities)
    # Removing the bottom 2% of low-density vertices usually cleans up artifacts
    mask = densities < np.quantile(densities, 0.02) 
    mesh.remove_vertices_by_mask(mask)
    return mesh

def create_mesh_surface(pcd):
    """
    Method 2: Ball Pivoting Algorithm (BPA)
    Rolls a virtual ball over the points. Only creates faces where points exist.
    """
    print("Mode: SURFACE (Ball Pivoting)")
    
    # 1. Calculate average distance between points to size the ball correctly
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    
    # 2. Define ball radii (try connecting points at 1x, 2x, and 4x distance)
    # These radii determine how large a hole the ball can bridge.
    radii = [avg_dist * 1, avg_dist * 2, avg_dist * 4]
    print(f"   -> Computed radii: {radii}")

    # 3. Create Mesh
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    
    return mesh

def main():
    # --- INPUT SETTINGS ---
    input_file = r"C:\Users\Tvang\Downloads\Result_Point_Cloud-20260128T065157Z-3-001\Result_Point_Cloud\Bear_28_01_cleaned_0.3.ply"  # Change this to your PLY filename
    output_stl = input_file.replace('.ply', '_') + MODE+'_final_mesh.stl'

    print(f"--- READING: {input_file} ---")
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    # 1. Load Data
    # Open3D handles PLY headers and binary/ascii parsing natively
    try:
        pcd = o3d.io.read_point_cloud(input_file)
    except Exception as e:
        print(f"Error reading PLY: {e}")
        sys.exit(1)

    # Verify points exist
    if not pcd.has_points():
        print("ERROR: PLY file is empty or could not be read.")
        sys.exit(1)
    
    print(f"Loaded {len(pcd.points)} points.")

    # 2. Estimate Normals
    # PLY files often contain normals already. If not, we estimate them.
    if not pcd.has_normals():
        print("No normals found in file. Estimating Normals...")
        # Note: 'radius' depends on your object scale. 
        # If your object is small (meters), 10 might be too big. 
        # If in mm, 10 is likely fine. Adjust 'radius=10' if results look odd.
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)
    else:
        print("Using existing normals from PLY file.")

    # 3. Run Selected Mode
    if MODE == "watertight":
        mesh = create_mesh_watertight(pcd)
    elif MODE == "surface":
        mesh = create_mesh_surface(pcd)
    else:
        print(f"Error: Unknown mode '{MODE}'")
        sys.exit(1)

    # 4. Finalize and Save
    if len(mesh.vertices) == 0:
        print("ERROR: Generated mesh is empty. Try adjusting parameters.")
        sys.exit(1)

    print("Computing Mesh Normals...")
    mesh.compute_vertex_normals()

    print(f"Saving to {output_stl}...")
    o3d.io.write_triangle_mesh(output_stl, mesh)
    print("SUCCESS.")

if __name__ == "__main__":
    main()