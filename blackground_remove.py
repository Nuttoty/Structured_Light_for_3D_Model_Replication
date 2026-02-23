import open3d as o3d

# Load the scan
pcd = o3d.io.read_point_cloud(r"C:\Users\Tvang\Downloads\budda09_02_13_cleaned.ply")

# 1. Segment the largest plane (the wall)
# distance_threshold: How far a point can be from the plane to be 'part' of the wall
# ransac_n: Number of points to sample for a plane (3 is standard)
# num_iterations: How many times to try finding the best plane
plane_model, inliers = pcd.segment_plane(distance_threshold=50,
                                         ransac_n=3,
                                         num_iterations=1000)

# 2. 'inliers' are the wall points. 'outliers' are your object.
wall_cloud = pcd.select_by_index(inliers)
object_cloud = pcd.select_by_index(inliers, invert=True)

# 3. Save the cleaned object
o3d.io.write_point_cloud(r"C:\Users\Tvang\Downloads\budda09_02_13_cleaned_wallremove.ply", object_cloud)

# Visualize the result (Wall in red, Object in original color)
wall_cloud.paint_uniform_color([1.0, 0, 0])
o3d.visualization.draw_geometries([object_cloud, wall_cloud])