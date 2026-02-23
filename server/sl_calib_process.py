import numpy as np
import cv2
import os
import glob
import scipy.io

# ================= CONFIGURATION =================
CHECKER_ROWS = 7        # Inner corners (e.g. standard 9x7 board has 8x6 inner corners)
CHECKER_COLS = 7
SQUARE_SIZE = 35.0      # Size of one square in mm
DATA_DIR = "17_02_2026_3Dscan\calib1"
SAVE_PATH = "17_02_2026_3Dscan\calib_result"
PROJ_WIDTH = 1920
PROJ_HEIGHT = 1080
# =================================================

def decode_gray(gray_stack, inv_stack, shape):
    # Pixel-wise decoding
    h, w, n = gray_stack.shape
    out = np.zeros((h, w), dtype=np.uint16)
    
    # Binary pattern
    binary = np.zeros((h, w, n), dtype=np.uint8)
    
    for i in range(n):
        bit = np.zeros((h, w), dtype=np.uint8)
        # Robust thresholding using inverse pattern
        bit[gray_stack[:,:,i] > inv_stack[:,:,i]] = 1
        binary[:,:,i] = bit

    # Binary to Decimal (Gray decoding)
    # Convert binary to standard binary first
    curr = np.zeros((h,w), dtype=np.uint8)
    for i in range(n):
        curr = np.bitwise_xor(curr, binary[:,:,i])
        out += (curr.astype(np.uint16) * (2**(n - 1 - i)))
        
    return out

def main():
    # Setup Checkerboard World Points
    objp = np.zeros((CHECKER_ROWS * CHECKER_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKER_ROWS, 0:CHECKER_COLS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # Storage
    obj_points = [] # 3d point in real world space
    cam_img_points = [] # 2d points in camera plane
    proj_img_points = [] # 2d points in projector plane

    poses = sorted(os.listdir(DATA_DIR))
    print(f"Found {len(poses)} poses. Processing...")

    for pose in poses:
        path = os.path.join(DATA_DIR, pose)
        if not os.path.isdir(path): continue
        
        # Load Texture Image (01.png)
        img = cv2.imread(os.path.join(path, "01.png"))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find Checkerboard
        ret, corners = cv2.findChessboardCorners(gray, (CHECKER_ROWS, CHECKER_COLS), None)
        
        if ret:
            # Subpixel refinement
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            # --- Projector Decoding at Corner Locations ---
            # Load Gray Code Images
            # Logic: We read the images, get the pixel values at 'corners2' coords
            # and reconstruct the projector column/row for those specific points.
            
            # Count patterns
            files = sorted(glob.glob(os.path.join(path, "*.png")))
            # Expect: 01, 02 (refs), then 2*cols, 2*rows
            num_pat_files = len(files) - 2
            n_col_bits = int(np.ceil(np.log2(PROJ_WIDTH)))
            n_row_bits = int(np.ceil(np.log2(PROJ_HEIGHT)))
            
            if num_pat_files != 2 * (n_col_bits + n_row_bits):
                print(f"Skipping {pose}: Image count mismatch.")
                continue

            # Decode Columns
            col_val = np.zeros(len(corners2))
            base_idx = 2
            for b in range(n_col_bits):
                img_p = cv2.imread(files[base_idx], 0)     # Normal
                img_i = cv2.imread(files[base_idx+1], 0)   # Inverse
                base_idx += 2
                
                # Sample intensities at corner locations
                # Using bilinear interpolation via remap or simpler mapping
                x = corners2[:,0,0]
                y = corners2[:,0,1]
                
                # Map coordinates to integers for direct indexing (simple method)
                # For better accuracy, use cv2.getRectSubPix or similar
                vals_p = img_p[y.astype(int), x.astype(int)]
                vals_i = img_i[y.astype(int), x.astype(int)]
                
                bit = (vals_p > vals_i).astype(int)
                
                # Gray code to Binary conversion happens iteratively? 
                # Actually, easier to store bits and decode full integer later.
                # Let's simplify: Just store the Gray bits
                if b == 0: 
                    gray_code = bit
                    bin_code = bit
                else:
                    gray_code = bit
                    bin_code = np.bitwise_xor(bin_code, bit) # Binary conversion
                
                col_val += bin_code * (2**(n_col_bits - 1 - b))

            # Decode Rows (similar logic)
            row_val = np.zeros(len(corners2))
            for b in range(n_row_bits):
                img_p = cv2.imread(files[base_idx], 0)
                img_i = cv2.imread(files[base_idx+1], 0)
                base_idx += 2
                vals_p = img_p[y.astype(int), x.astype(int)]
                vals_i = img_i[y.astype(int), x.astype(int)]
                bit = (vals_p > vals_i).astype(int)
                
                if b == 0: bin_code = bit
                else: bin_code = np.bitwise_xor(bin_code, bit)
                
                row_val += bin_code * (2**(n_row_bits - 1 - b))

            # Valid points?
            # Create Projector "Image Points"
            proj_pts = np.column_stack((col_val, row_val)).astype(np.float32)
            
            obj_points.append(objp)
            cam_img_points.append(corners2)
            proj_img_points.append(proj_pts.reshape(-1, 1, 2))
            print(f"  Processed {pose}")

    print("Calibrating Camera...")
    ret_c, mtx_c, dist_c, rvecs_c, tvecs_c = cv2.calibrateCamera(obj_points, cam_img_points, (img.shape[1], img.shape[0]), None, None)
    
    print("Calibrating Projector...")
    ret_p, mtx_p, dist_p, rvecs_p, tvecs_p = cv2.calibrateCamera(obj_points, proj_img_points, (PROJ_WIDTH, PROJ_HEIGHT), None, None)

    print("Stereo Calibration...")
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        obj_points, cam_img_points, proj_img_points, 
        mtx_c, dist_c, mtx_p, dist_p, (img.shape[1], img.shape[0]),
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    print(f"Calibration Done. Reprojection Error: {retval}")

    # ================= GENERATE MATLAB COMPATIBLE OUTPUT =================
    # We need: Nc (Camera Rays), Oc (Camera Center), wPlaneCol, wPlaneRow
    
    # 1. Camera Center (Oc)
    # In Camera frame, Camera is at [0,0,0]
    Oc = np.zeros((3, 1)) 

    # 2. Camera Rays (Nc)
    # Nc is the ray vector for every pixel in the camera image
    # Inverse project every pixel (u,v) -> (x,y,1) -> normalize
    h, w = img.shape[:2]
    # Meshgrid of pixels
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    # Un-distort points? 
    # For 'Nc' in standard structured light, we usually assume a pinhole model 
    # Or we store the un-distorted rays.
    # Simple Pinhole approximation:
    fx, fy = cameraMatrix1[0,0], cameraMatrix1[1,1]
    cx, cy = cameraMatrix1[0,2], cameraMatrix1[1,2]
    
    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy
    z_norm = np.ones_like(x_norm)
    
    rays = np.stack((x_norm, y_norm, z_norm), axis=2) # (H, W, 3)
    # Normalize
    norms = np.linalg.norm(rays, axis=2, keepdims=True)
    rays /= norms
    # Reshape to (3, H*W) for Matlab
    Nc = rays.reshape(-1, 3).T 

    # 3. Projector Planes (wPlaneCol / wPlaneRow)
    # We need the plane equation for every column and row of the projector
    # in the CAMERA coordinate system.
    
    # Projector Origin in Cam Frame: T
    # Projector Rotation: R
    Proj_Center = T # (3, 1)

    # For each column C in projector:
    # It forms a plane with the Projector Center and the vertical line at C in Proj Image.
    # We need 2 points on that line in Proj Image Space, transform to Cam Space, then fit plane.
    
    wPlaneCol = np.zeros((PROJ_WIDTH, 4))
    wPlaneRow = np.zeros((PROJ_HEIGHT, 4))
    
    # Invert Proj Matrix for ray lookup
    fx_p, fy_p = cameraMatrix2[0,0], cameraMatrix2[1,1]
    cx_p, cy_p = cameraMatrix2[0,2], cameraMatrix2[1,2]

    # Pre-calculate rotation from Proj to Cam (R is Proj relative to Cam? Standard stereo is P2 = R*P1 + T)
    # So P_proj = R * P_cam + T  =>  P_cam = R.T * (P_proj - T)
    # Actually stereoCalibrate returns R, T that transforms points from Cam1 to Cam2 (Proj).
    # X_proj = R * X_cam + T.
    # So X_cam = R^T * (X_proj - T)
    R_inv = R.T
    
    # Origin of Proj in Cam Frame
    # X_proj = [0,0,0] => X_cam = -R^T * T
    C_p_cam = -R_inv @ T

    def get_plane_from_proj_line(u_p, v_p_start, v_p_end, is_col=True):
        # 1. Point in Proj Image (Normalized)
        if is_col:
            p1_n = np.array([(u_p - cx_p)/fx_p, (v_p_start - cy_p)/fy_p, 1]).reshape(3,1)
            p2_n = np.array([(u_p - cx_p)/fx_p, (v_p_end - cy_p)/fy_p, 1]).reshape(3,1)
        else:
            p1_n = np.array([(v_p_start - cx_p)/fx_p, (u_p - cy_p)/fy_p, 1]).reshape(3,1) # u is y here
            p2_n = np.array([(v_p_end - cx_p)/fx_p, (u_p - cy_p)/fy_p, 1]).reshape(3,1)

        # 2. Rotate to Camera Frame (Directions only)
        # Ray direction in Cam frame = R^T * Ray_proj
        r1 = R_inv @ p1_n
        r2 = R_inv @ p2_n
        
        # 3. Form Plane
        # Plane contains C_p_cam, (C_p_cam + r1), (C_p_cam + r2)
        # Normal is cross product of r1 and r2? No.
        # It's plane passing through C_p_cam spanning vectors r1 and r2?
        # Actually for a column, r1 and r2 are along the same vertical line.
        # Normal = Cross(r1, r2) ? No, they might be collinear with origin.
        # Normal is Cross(r1, r2 - r1)? 
        
        # Better: Cross product of vector (C_p + r1 - C_p) and (C_p + r2 - C_p) = Cross(r1, r2)
        normal = np.cross(r1.flatten(), r2.flatten())
        normal /= np.linalg.norm(normal)
        
        # d = -normal . point (use C_p_cam)
        d = -np.dot(normal, C_p_cam.flatten())
        
        return np.array([normal[0], normal[1], normal[2], d])

    for c in range(PROJ_WIDTH):
        wPlaneCol[c, :] = get_plane_from_proj_line(c, 0, PROJ_HEIGHT, is_col=True)
        
    for r in range(PROJ_HEIGHT):
        wPlaneRow[r, :] = get_plane_from_proj_line(r, 0, PROJ_WIDTH, is_col=False)

    # Save
    if not os.path.exists(os.path.dirname(SAVE_PATH)):
        os.makedirs(os.path.dirname(SAVE_PATH))
        
    scipy.io.savemat(SAVE_PATH, {
        "Nc": Nc,
        "Oc": Oc,
        "wPlaneCol": wPlaneCol.T, # Matlab expects (4, N) usually? Check slScan.
        "wPlaneRow": wPlaneRow.T,
        "cam_K": cameraMatrix1,
        "proj_K": cameraMatrix2,
        "R": R,
        "T": T
    })
    print(f"Saved {SAVE_PATH}")

if __name__ == "__main__":
    main()