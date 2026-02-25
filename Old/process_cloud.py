import numpy as np
import cv2
import scipy.io
import os
import argparse
import glob

def load_calibration(calib_path):
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Calibration file not found at {calib_path}")
    
    print(f"Loading calibration from {calib_path}...")
    data = scipy.io.loadmat(calib_path)
    
    # Extract needed variables
    # Note: Matlab often saves as 2D arrays, we flatten where appropriate
    return {
        "Nc": data["Nc"],               # Camera Rays (3, N_pixels)
        "Oc": data["Oc"],               # Camera Center (3, 1)
        "wPlaneCol": data["wPlaneCol"], # Projector Col Planes (4, Proj_W)
        "wPlaneRow": data["wPlaneRow"], # Projector Row Planes (4, Proj_H)
        "cam_K": data["cam_K"]          # Camera Matrix
    }

def gray_decode(folder, n_cols=1920, n_rows=1080):
    """
    Decodes Gray Code pattern images into decimal column/row maps.
    Uses float32 to avoid 'uint8' overflow errors.
    """
    # 1. Sort files naturally
    files = sorted(glob.glob(os.path.join(folder, "*.bmp")))
    if not files:
        files = sorted(glob.glob(os.path.join(folder, "*.png")))
        
    if len(files) < 4:
        raise ValueError("Not enough images in folder to decode.")

    # 2. Read Reference Images
    # We assume the standard order: 01 (White), 02 (Black), then Pattern/Inverse pairs
    img_white = cv2.imread(files[0], 0).astype(np.float32)
    img_black = cv2.imread(files[1], 0).astype(np.float32)
    
    height, width = img_white.shape
    
    # Shadow Mask: Ignore pixels that are too dark in the white image
    # or where the contrast between black/white is too low
    mask_shadow = img_white > 40
    mask_contrast = (img_white - img_black) > 10
    valid_mask = mask_shadow & mask_contrast

    # 3. Decode Loop
    # Calculate number of bits needed
    n_col_bits = int(np.ceil(np.log2(n_cols)))
    n_row_bits = int(np.ceil(np.log2(n_rows)))
    
    total_patterns = (n_col_bits + n_row_bits) * 2
    if len(files) - 2 < total_patterns:
        print(f"Warning: Expected {total_patterns} pattern files, found {len(files)-2}")

    current_idx = 2
    
    def decode_sequence(n_bits):
        nonlocal current_idx
        
        # Use int32 to safely store values > 255 (up to 2 billion)
        gray_val = np.zeros((height, width), dtype=np.int32)
        binary_val = np.zeros((height, width), dtype=np.int32)
        
        for b in range(n_bits):
            if current_idx >= len(files): break
            
            p_path = files[current_idx]; current_idx += 1
            i_path = files[current_idx]; current_idx += 1
            
            img_p = cv2.imread(p_path, 0).astype(np.float32)
            img_i = cv2.imread(i_path, 0).astype(np.float32)
            
            # Threshold: Is Pattern > Inverse?
            bit = np.zeros((height, width), dtype=np.int32)
            bit[img_p > img_i] = 1
            
            # Standard Gray Code to Binary conversion:
            # Binary[0] = Gray[0]
            # Binary[i] = Binary[i-1] XOR Gray[i]
            # But here we are decoding bit-planes directly.
            # Simplified approach: Gray Code bits -> Decimal Gray -> Decimal Binary
            
            # Actually, standard approach is easier:
            # Store the bits, then convert the full integer from Gray to Binary
            gray_val = np.bitwise_or(gray_val, np.left_shift(bit, (n_bits - 1 - b)))

        # Convert Gray Integer to Binary Integer
        # https://en.wikipedia.org/wiki/Gray_code#Converting_to_and_from_Gray_code
        mask = np.right_shift(gray_val, 1)
        while np.any(mask > 0):
            gray_val = np.bitwise_xor(gray_val, mask)
            mask = np.right_shift(mask, 1)
            
        return gray_val

    print("Decoding Columns...")
    col_map = decode_sequence(n_col_bits)
    
    print("Decoding Rows...")
    row_map = decode_sequence(n_row_bits)
    
    return col_map, row_map, valid_mask, cv2.imread(files[0]) # Return texture too

def reconstruct_point_cloud(col_map, row_map, mask, texture, calib):
    print("Reconstructing 3D points...")
    
    Nc = calib["Nc"] # (3, H*W) usually or (3, N)
    Oc = calib["Oc"] # (3, 1)
    wPlaneCol = calib["wPlaneCol"] # (4, N) or (N, 4) depending on save
    
    # Handle Transpose issues from Matlab saving
    if wPlaneCol.shape[0] == 4: wPlaneCol = wPlaneCol.T
    
    h, w = col_map.shape
    
    # Create lists for valid points
    points = []
    colors = []
    
    # We iterate only valid pixels to save memory/time
    # (Vectorization is faster but complex to explain; keeping loop for clarity or partial vectorization)
    
    # Flatten arrays
    col_flat = col_map.flatten()
    mask_flat = mask.flatten()
    tex_flat = texture.reshape(-1, 3) # BGR
    
    # Indices where mask is true
    valid_indices = np.where(mask_flat)[0]
    
    print(f"Processing {len(valid_indices)} valid pixels...")
    
    # Pre-fetch Rays for valid pixels
    # Nc might be shape (3, H*W)
    if Nc.shape[1] == h * w:
        rays = Nc[:, valid_indices] # (3, N_valid)
    else:
        # If Nc was saved differently, we might need to regenerate it.
        # Let's assume standard pinhole regeneration if Nc seems wrong.
        # (Implementing Ray gen just in case)
        K = calib["cam_K"]
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        
        y_v, x_v = np.unravel_index(valid_indices, (h, w))
        x_n = (x_v - cx) / fx
        y_n = (y_v - cy) / fy
        z_n = np.ones_like(x_n)
        
        rays = np.stack((x_n, y_n, z_n)) # (3, N)
        norms = np.linalg.norm(rays, axis=0)
        rays /= norms
        
    # Get Projector Column Indices
    proj_cols = col_flat[valid_indices]
    
    # Clip to ensure we don't index out of bounds of the planes array
    proj_cols = np.clip(proj_cols, 0, wPlaneCol.shape[0] - 1)
    
    # Gather Planes
    # wPlaneCol[c] = [nx, ny, nz, d]
    planes = wPlaneCol[proj_cols, :] # (N_valid, 4)
    
    # Ray-Plane Intersection
    # P = Oc + t * R
    # N dot P + d = 0
    # N dot (Oc + t*R) + d = 0
    # t * (N dot R) + (N dot Oc + d) = 0
    # t = - (N dot Oc + d) / (N dot R)
    
    # N is planes[:, 0:3], d is planes[:, 3]
    N = planes[:, 0:3].T # (3, N)
    d = planes[:, 3]     # (N,)
    
    # Dot Products
    # Oc is (3,1)
    denom = np.sum(N * rays, axis=0) # Dot product of Normal and Ray
    numer = np.dot(N.T, Oc).flatten() + d # Dot product Normal and Origin + d
    
    # Avoid division by zero
    valid_intersect = np.abs(denom) > 1e-6
    
    t = -numer[valid_intersect] / denom[valid_intersect]
    
    # Calculate Points
    # P = Oc + t * R
    rays_valid = rays[:, valid_intersect]
    P = Oc + rays_valid * t
    
    # Colors
    C = tex_flat[valid_indices[valid_intersect]]
    
    return P.T, C

def save_ply(points, colors, filename):
    print(f"Saving {len(points)} points to {filename}...")
    
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(points)):
            p = points[i]
            c = colors[i]
            # PLY expects RGB, OpenCV is BGR
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[2]} {c[1]} {c[0]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode and Reconstruct 3D Scan")
    parser.add_argument("--input", required=True, help="Folder containing scan images")
    parser.add_argument("--output", default="output.ply", help="Output .ply file")
    parser.add_argument("--calib", default="./calib/calib_results/calib_cam_proj.mat", help="Path to calibration mat file")
    
    args = parser.parse_args()
    
    try:
        calib_data = load_calibration(args.calib)
        c_map, r_map, mask, texture = gray_decode(args.input)
        points, colors = reconstruct_point_cloud(c_map, r_map, mask, texture, calib_data)
        save_ply(points, colors, args.output)
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")