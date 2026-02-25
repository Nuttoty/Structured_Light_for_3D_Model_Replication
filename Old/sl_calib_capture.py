import numpy as np
import cv2
import os
import time

# ================= CONFIGURATION =================
# Hardware
SCREEN_OFFSET_X = 1920  # Width of main monitor
PROJ_WIDTH = 1024
PROJ_HEIGHT = 768
CAM_ID = 0              # Camera Index
CAM_WIDTH = 1600
CAM_HEIGHT = 1200

# Calibration Settings
NUM_POSES = 10          # Number of different board positions to capture
Gray_Pattern_Scale = 1  # 1 = Pixel Perfect
output_dir = "./calib_data"
# =================================================

def generate_gray_code_patterns(width, height):
    # Same generator as before
    n_cols = int(np.ceil(np.log2(width)))
    n_rows = int(np.ceil(np.log2(height)))
    def get_gray_1d(n):
        if n == 1: return ['0', '1']
        prev = get_gray_1d(n - 1)
        return ['0' + s for s in prev] + ['1' + s for s in prev[::-1]]
    
    col_gray = get_gray_1d(n_cols)
    row_gray = get_gray_1d(n_rows)
    P = [[], []]
    
    for b in range(n_cols):
        pat = np.zeros((height, width), dtype=np.uint8)
        for c in range(width):
             if c < len(col_gray) and col_gray[c][b] == '1': pat[:, c] = 255
        P[0].append(pat)
    for b in range(n_rows):
        pat = np.zeros((height, width), dtype=np.uint8)
        for r in range(height):
            if r < len(row_gray) and row_gray[r][b] == '1': pat[r, :] = 255
        P[1].append(pat)
    return P

def main():
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # Setup Window
    window_name = "Calibration Projector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, SCREEN_OFFSET_X, 0)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Setup Camera
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    # Generate Patterns
    P = generate_gray_code_patterns(PROJ_WIDTH, PROJ_HEIGHT)
    white_img = np.full((PROJ_HEIGHT, PROJ_WIDTH), 255, dtype=np.uint8)
    black_img = np.zeros((PROJ_HEIGHT, PROJ_WIDTH), dtype=np.uint8)

    print(f"=== Starting Calibration Capture ({NUM_POSES} Poses) ===")
    print("1. Hold your printed checkerboard in view.")
    print("2. Press ENTER to capture a pose.")
    print("3. Move board to new angle/position.")
    print("4. Repeat.")

    for pose_i in range(1, NUM_POSES + 1):
        pose_dir = os.path.join(output_dir, f"pose_{pose_i:02d}")
        if not os.path.exists(pose_dir): os.makedirs(pose_dir)

        # 1. Waiting Loop (Project White)
        while True:
            cv2.imshow(window_name, white_img)
            ret, frame = cap.read()
            # Draw text on frame
            disp_frame = frame.copy()
            cv2.putText(disp_frame, f"Pose {pose_i}/{NUM_POSES} - Press ENTER", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Camera View", disp_frame)
            if cv2.waitKey(1) == 13: # Enter key
                break
        
        print(f"Capturing Pose {pose_i}...")

        # 2. Capture Reference (White/Black)
        # Capture White (Texture)
        cv2.imshow(window_name, white_img)
        cv2.waitKey(500)
        ret, frame = cap.read()
        cv2.imwrite(os.path.join(pose_dir, "01.png"), frame) # 01 is texture ref

        # Capture Black
        cv2.imshow(window_name, black_img)
        cv2.waitKey(200)
        ret, frame = cap.read()
        cv2.imwrite(os.path.join(pose_dir, "02.png"), frame) 

        # 3. Capture Gray Codes
        img_idx = 3
        for j in range(2): # Cols, Rows
            for pat in P[j]:
                # Normal
                cv2.imshow(window_name, pat)
                cv2.waitKey(200)
                _, frame = cap.read()
                cv2.imwrite(os.path.join(pose_dir, f"{img_idx:02d}.png"), frame)
                img_idx += 1
                
                # Inverse
                cv2.imshow(window_name, 255 - pat)
                cv2.waitKey(200)
                _, frame = cap.read()
                cv2.imwrite(os.path.join(pose_dir, f"{img_idx:02d}.png"), frame)
                img_idx += 1
        
        print(f"Pose {pose_i} saved.")

    cv2.destroyAllWindows()
    cap.release()
    print("Capture complete. Now run sl_calib_process.py")

if __name__ == "__main__":
    main()