"""
Read calibration from calib_cam_proj.mat (or .file) and print human-readable summary.
Supports .mat (MATLAB) files; focal length, rotation, and translation are reported.
"""
# python read_calib.py "C:\Users\Tvang\Downloads\Temp\calib_cam_proj.mat"
import argparse
import os
import sys
import numpy as np


def rotation_matrix_to_euler_degrees(R):
    """Convert 3x3 rotation matrix to Euler angles (degrees): roll, pitch, yaw (XYZ)."""
    try:
        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(R)
        angles = r.as_euler("xyz", degrees=True)
        return angles  # (roll, pitch, yaw) in degrees
    except ImportError:
        return None


def load_and_print_calib(calib_path):
    if not os.path.exists(calib_path):
        print(f"Error: File not found: {calib_path}")
        return False

    ext = os.path.splitext(calib_path)[1].lower()
    if ext == ".mat":
        try:
            import scipy.io
            data = scipy.io.loadmat(calib_path)
        except ImportError:
            print("Error: scipy is required for .mat files. Install with: pip install scipy")
            return False
    else:
        print("Error: Only .mat files are supported. Use calib_cam_proj.mat")
        return False

    # Remove MATLAB metadata keys
    keys = [k for k in data.keys() if not k.startswith("__")]
    print("=" * 60)
    print("Calibration summary (readable)")
    print("=" * 60)

    # ----- Camera intrinsics (cam_K) -----
    if "cam_K" in data:
        K = np.asarray(data["cam_K"])
        if K.shape == (3, 3):
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            print("\n--- Camera intrinsics ---")
            print(f"  Focal length (cam):  fx = {fx:.4f},  fy = {fy:.4f}")
            print(f"  Principal point:     cx = {cx:.4f},  cy = {cy:.4f}")
        else:
            print("\n  cam_K present but unexpected shape:", K.shape)
    else:
        print("\n  cam_K not found in file.")

    # ----- Projector intrinsics (proj_K) -----
    if "proj_K" in data:
        Kp = np.asarray(data["proj_K"])
        if Kp.shape == (3, 3):
            fpx, fpy = float(Kp[0, 0]), float(Kp[1, 1])
            cpx, cpy = float(Kp[0, 2]), float(Kp[1, 2])
            print("\n--- Projector intrinsics ---")
            print(f"  Focal length (proj): fx = {fpx:.4f},  fy = {fpy:.4f}")
            print(f"  Principal point:     cx = {cpx:.4f},  cy = {cpy:.4f}")
        else:
            print("\n  proj_K present but unexpected shape:", Kp.shape)

    # ----- Rotation (R) -----
    if "R" in data:
        R = np.asarray(data["R"])
        if R.shape == (3, 3):
            print("\n--- Rotation ---")
            print("  Rotation matrix R (3x3):")
            for i in range(3):
                print(f"    [{R[i,0]:10.5f}  {R[i,1]:10.5f}  {R[i,2]:10.5f}]")
            angles = rotation_matrix_to_euler_degrees(R)
            if angles is not None:
                roll, pitch, yaw = angles
                print(f"  Rotation angles (deg, XYZ): roll = {roll:.4f},  pitch = {pitch:.4f},  yaw = {yaw:.4f}")
        else:
            print("\n  R present but unexpected shape:", R.shape)
    else:
        print("\n  R (rotation) not found in file.")

    # ----- Translation (T) -----
    if "T" in data:
        T = np.asarray(data["T"]).flatten()
        if T.size >= 3:
            print("\n--- Translation ---")
            print(f"  Translation T:  tx = {T[0]:.6f},  ty = {T[1]:.6f},  tz = {T[2]:.6f}")
        else:
            print("\n  T present but unexpected size:", T.size)
    else:
        print("\n  T (translation) not found in file.")

    # ----- Camera center (Oc) -----
    if "Oc" in data:
        Oc = np.asarray(data["Oc"]).flatten()
        if Oc.size >= 3:
            print("\n--- Camera center (world) ---")
            print(f"  Oc:  x = {Oc[0]:.6f},  y = {Oc[1]:.6f},  z = {Oc[2]:.6f}")
        else:
            print("\n  Oc present but unexpected size:", Oc.size)

    print("\n" + "=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Translate calib_cam_proj.mat (or .file) into readable text (focal length, rotation, translation)."
    )
    parser.add_argument(
        "calib_file",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__) or ".", "calib_cam_proj.mat"),
        help="Path to calibration file (default: calib_cam_proj.mat in same folder)",
    )
    args = parser.parse_args()
    ok = load_and_print_calib(args.calib_file)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
