import os
import sys
import threading
import uuid

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from android_camera_host_client import AndroidCameraHostClient, CameraSettings

# ==========================================
# CONFIGURATION (Projector / Patterns)
# ==========================================
SCREEN_OFFSET_X = 1920  # Set this to your primary monitor width
SCREEN_WIDTH = 1024     # Projector Width
SCREEN_HEIGHT = 768     # Projector Height
PROJ_VALUE = 200        # Pattern Brightness (0-255)
DATA_DIR = "./data"
D_SAMPLE_PROJ = 1       # Downsampling factor for pattern generation

# Camera host connection:
# - Wi-Fi:   "http://PHONE_IP:8765"
# - USB ADB: "http://127.0.0.1:8765" after `adb reverse tcp:8765 tcp:8765`
DEFAULT_CAMERA_HOST_URL = "http://127.0.0.1:8765"


class SLSystem:
    def __init__(self, root, client: AndroidCameraHostClient):
        self.root = root
        self.client = client
        self.window_name = "Projector"

        # Structured-light-friendly defaults (best-effort):
        # Lock exposure+focus to reduce flicker and keep patterns stable.
        self.capture_settings = CameraSettings(
            jpeg_quality=95,
            ae_mode="off",
            af_mode="off",
            awb_mode="auto",
            eis=False,
        )

    def init_projector_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window_name, SCREEN_OFFSET_X, 0)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        black = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8)
        cv2.imshow(self.window_name, black)
        cv2.waitKey(10)

    def close_projector_window(self):
        cv2.destroyWindow(self.window_name)

    def generate_patterns(self):
        width, height = SCREEN_WIDTH // D_SAMPLE_PROJ, SCREEN_HEIGHT // D_SAMPLE_PROJ
        n_cols = int(np.ceil(np.log2(width)))
        n_rows = int(np.ceil(np.log2(height)))

        def get_gray_1d(n):
            if n == 1:
                return ["0", "1"]
            prev = get_gray_1d(n - 1)
            return ["0" + s for s in prev] + ["1" + s for s in prev[::-1]]

        col_gray = get_gray_1d(n_cols)
        row_gray = get_gray_1d(n_rows)

        P = [[], []]  # [Vertical_Stripes, Horizontal_Stripes]

        for b in range(n_cols):
            pat = np.zeros((height, width), dtype=np.uint8)
            for c in range(width):
                if c < len(col_gray) and col_gray[c][b] == "1":
                    pat[:, c] = 1
            P[0].append(pat)

        for b in range(n_rows):
            pat = np.zeros((height, width), dtype=np.uint8)
            for r in range(height):
                if r < len(row_gray) and row_gray[r][b] == "1":
                    pat[r, :] = 1
            P[1].append(pat)

        return P

    def trigger_phone_capture(self, save_path: str) -> bool:
        """
        Calls the Android camera host and saves JPEG bytes directly.
        """
        try:
            meta = self.client.capture_to_path(save_path, settings=self.capture_settings)
            print(f"[Phone] Saved {os.path.basename(save_path)} meta={meta}")
            return True
        except Exception as e:
            print(f"[Phone] Capture error: {e}")
            return False

    def run_scan_sequence(self, obj_name: str):
        self.init_projector_window()
        P = self.generate_patterns()

        scan_dir = os.path.join(DATA_DIR, "scans", obj_name)
        os.makedirs(scan_dir, exist_ok=True)

        patterns = []
        white = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8) * PROJ_VALUE
        black = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8)

        patterns.append(("01.jpg", white))
        patterns.append(("02.jpg", black))

        idx = 3
        for j in range(2):
            for pat in P[j]:
                pat_img = (pat * PROJ_VALUE).astype(np.uint8)
                inv_img = ((1 - pat) * PROJ_VALUE).astype(np.uint8)
                if D_SAMPLE_PROJ > 1:
                    pat_img = cv2.resize(pat_img, (SCREEN_WIDTH, SCREEN_HEIGHT), interpolation=cv2.INTER_NEAREST)
                    inv_img = cv2.resize(inv_img, (SCREEN_WIDTH, SCREEN_HEIGHT), interpolation=cv2.INTER_NEAREST)

                patterns.append((f"{idx:02d}.jpg", pat_img))
                idx += 1
                patterns.append((f"{idx:02d}.jpg", inv_img))
                idx += 1

        cv2.imshow(self.window_name, white)
        cv2.waitKey(10)
        messagebox.showinfo("Scan", f"Ready to scan '{obj_name}'.\nClick OK to start.")

        for fname, img in patterns:
            cv2.imshow(self.window_name, img)
            cv2.waitKey(200)

            save_path = os.path.join(scan_dir, fname)
            success = self.trigger_phone_capture(save_path)
            if not success:
                messagebox.showerror("Error", "Phone capture failed (check connection/app).")
                self.close_projector_window()
                return

        self.close_projector_window()
        messagebox.showinfo("Done", f"Scan captured to:\n{scan_dir}")


class ScannerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Structured Light Controller (Android Camera Host)")
        self.root.geometry("520x360")

        self.url_var = tk.StringVar(value=DEFAULT_CAMERA_HOST_URL)
        self.status_var = tk.StringVar(value="Not connected")
        self.obj_name_var = tk.StringVar(value="test_object")

        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")

        ttk.Label(root, text="Android Camera Host Controller", font=("Arial", 16)).pack(pady=15)

        frm_conn = ttk.Frame(root)
        frm_conn.pack(fill=tk.X, padx=15, pady=10)

        ttk.Label(frm_conn, text="Phone Base URL:").pack(anchor="w")
        ttk.Entry(frm_conn, textvariable=self.url_var).pack(fill=tk.X, pady=5)

        btns = ttk.Frame(frm_conn)
        btns.pack(fill=tk.X, pady=5)
        ttk.Button(btns, text="Test Connection", command=self.test_connection).pack(side=tk.LEFT)
        ttk.Label(btns, textvariable=self.status_var, foreground="blue").pack(side=tk.LEFT, padx=10)

        frm_obj = ttk.Frame(root)
        frm_obj.pack(fill=tk.X, padx=15, pady=10)
        ttk.Label(frm_obj, text="Object Name:").pack(anchor="w")
        ttk.Entry(frm_obj, textvariable=self.obj_name_var).pack(fill=tk.X, pady=5)

        ttk.Button(root, text="Start Scan", command=self.start_scan).pack(fill=tk.X, padx=15, pady=20)

        ttk.Label(
            root,
            text="USB tip: run `adb reverse tcp:8765 tcp:8765` then use http://127.0.0.1:8765",
            foreground="gray",
            wraplength=480,
        ).pack(padx=15, pady=10)

        self.client: AndroidCameraHostClient | None = None

    def test_connection(self):
        try:
            self.client = AndroidCameraHostClient(self.url_var.get().strip())
            st = self.client.ping()
            self.status_var.set(f"OK: {st.get('device')} cam={st.get('activeCameraId')}")
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            self.client = None

    def start_scan(self):
        if self.client is None:
            self.test_connection()
            if self.client is None:
                messagebox.showerror("Not connected", "Cannot connect to Android Camera Host.")
                return

        name = self.obj_name_var.get().strip()
        if not name:
            return

        sl = SLSystem(self.root, self.client)
        threading.Thread(target=sl.run_scan_sequence, args=(name,), daemon=True).start()


def main():
    root = tk.Tk()
    app = ScannerGUI(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()

