import os
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog

from config import DEFAULT_ROOT
from sl_system import SLSystem
from processing import ProcessingLogic
from arduino import ArduinoController

# ==========================================
# GUI (Graphical User Interface)
# ==========================================
class ScannerGUI:
    # คลาสสำหรับสร้างหน้าต่างโปรแกรมผู้ใช้งาน (UI) ด้วย Tkinter
    def __init__(self, root):
        self.root = root
        self.root.title("Project 3D Scanner Suite") # ตั้งชื่อหน้าต่างโปรแกรม
        self.root.geometry("800x700") # ขนาดเริ่มต้น กว้าง 800 สูง 700 พิเซล
        
        # ผูกยึดสคริปต์การทำงานของแต่ละส่วนเข้ากับตัวแปรในหน้าต่าง GUI นี้
        self.sys = SLSystem()             # ระบบควบคุมแสงและสแกน
        self.processor = ProcessingLogic() # ระบบคัดกรองกราฟิก 3D (Open3D)
        self.arduino = ArduinoController() # ระบบหมุนมอเตอร์ผ่าน Arduino
        
        # --- State Variables (Scanner) ---
        # ตัวแปรจำพวก StringVar, IntVar เอาไว้ผูกค่าทิ้งไว้ แล้วดึงลิงก์ไปแสดงบนหน้าจอ ช่องกรอกได้เรื่อยๆตลอดโปรแกรมรัน
        
        # ตั้งค่าช่องเก็บบันทึกไฟล์ภาพ Calibrate เริ่มต้น
        self.calib_capture_dir = tk.StringVar(value=os.path.join(DEFAULT_ROOT, "calib"))
        # พาธไฟล์ .mat ที่ได้จากการ Calibrate เสร็จแล้ว
        self.calib_file = tk.StringVar(value=os.path.join(DEFAULT_ROOT, "calib", "calib.mat"))
        # จำนวนท่าทางที่จะบังคับกระดานหมากรุก (เริ่มต้นให้เป็น 5 ท่า)
        self.num_poses = tk.IntVar(value=5)
        
        # ชื่อออบเจ็กต์หรือชิ้นงานที่จะนำมาสแกน (ใช้ตั้งเป็นชื่อโฟลเดอร์ตระกูลนี้ด้วย)
        self.scan_name = tk.StringVar(value="object_01")
        # โฟลเดอร์ปลายทางที่รูปภาพสแกนแต่ละบิตจะกระจัดกระจายหล่นไป
        self.scan_capture_dir = tk.StringVar(value=os.path.join(DEFAULT_ROOT, "scans", "object_01"))
        
        # --- State Variables (Combined Processing) ---
        # เมนูรวมพลสำหรับประมวลผล (Processing)
        self.proc_input_dir = tk.StringVar()  # โฟลเดอร์ขาเข้า
        self.proc_output_dir = tk.StringVar() # โฟลเดอร์ปลายทางออก
        
        # BG Params (พารามิเตอร์ลบพื้นหลัง)
        self.bg_dist_thresh = tk.DoubleVar(value=50.0) # ระยะห่างความลึกขอกำแพง
        self.bg_ransac_n = tk.IntVar(value=3) # จำนวนพอยต์สุ่ม
        self.bg_iterations = tk.IntVar(value=1000) # ความพยายามสุ่ม
        
        # Outlier Params (พารามิเตอร์ฝุ่นผง)
        self.proc_nb_neighbors = tk.IntVar(value=20) # จำนวนเพื่อนข้างเคียง
        self.proc_std_ratio = tk.DoubleVar(value=2.0) # เปอร์เซ็นต์ความห่าง
        
        # 360 Merge Params (พารามิเตอร์เย็บต่อโมเดล 360 องศา)
        self.merge_input_dir = tk.StringVar()
        self.merge_output_file = tk.StringVar()
        self.merge_voxel = tk.DoubleVar(value=0.02) # ความละเอียดขนาดบล็อคหยาบๆ
        
        # 360 Meshing Params (พารามิเตอร์ถักทอตาข่าย Mesh รอบตัว)
        self.m360_input_ply = tk.StringVar()
        self.m360_output_stl = tk.StringVar()
        self.m360_depth = tk.IntVar(value=10) # ความลึกสมการถักตะแกรง
        self.m360_trim = tk.DoubleVar(value=0.0) # ระดับการตัดเศษที่ลอกร่อน (Default to 0.0=Watertight)
        self.m360_mode = tk.StringVar(value="radial") # โหมดเล็งทิศทาง Normal (Default to Radial)

        # STL Reconstruction (พารามิเตอร์ปั้น 3D ฝั่งปกติ)
        self.s_input_ply = tk.StringVar()
        self.s_output_stl = tk.StringVar()
        self.s_mode = tk.StringVar(value="watertight")
        self.s_depth = tk.IntVar(value=10)
        self.s_radii = tk.StringVar(value="1, 2, 4")

        # --- State Variables (Turntable) ---
        # แท่นหมุน
        self.tt_port = tk.StringVar() # ช่องสำหรับบอกว่าอยู่ COM ไหน
        self.tt_baud = tk.StringVar(value="115200") # เสาความเร็วเชื่อมต่อ
        self.tt_degrees = tk.DoubleVar(value=30.0) # จำนวนองศาที่อยากให้หมุนแต่ละรอบ (เช่น 30 องศา)
        self.tt_turns = tk.IntVar(value=12) # จำนวนที่ต้องถ่าย (12 รอบ x 30 = 360)
        self.tt_status = tk.StringVar(value="Status: Idle")
        self.tt_base_name = tk.StringVar(value="Object_360")
        self.tt_save_dir = tk.StringVar(value=os.path.join(DEFAULT_ROOT, "scans_360"))

        # --- TABS (ตั้งค่าแท็บแต่ละแผ่นของโปรแกรม) ---
        self.notebook = ttk.Notebook(root) # สร้างหน้าต่างแฟ้มเมนูแนวนอน
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # คลอดพื้นที่สำหรับแต่ละหน้าแท็บ 6 รายการ
        self.tab_scan = ttk.Frame(self.notebook)
        self.tab_proc = ttk.Frame(self.notebook)
        self.tab_merge = ttk.Frame(self.notebook)
        self.tab_mesh360 = ttk.Frame(self.notebook) 
        self.tab_turntable = ttk.Frame(self.notebook) 
        self.tab_recon = ttk.Frame(self.notebook)
        
        # ยัดเฟรมเข้าไปในแฟ้มเมนู ให้มีหัวข้อ 1-6 ให้คลิกเปลี่ยนเรื่อง
        self.notebook.add(self.tab_scan, text="1. Scan & Generate")
        self.notebook.add(self.tab_proc, text="2. Cleanup & Process")
        self.notebook.add(self.tab_merge, text="3. Merge 360")
        self.notebook.add(self.tab_mesh360, text="4. 360 Meshing")
        self.notebook.add(self.tab_turntable, text="5. Auto-Scan 360")
        self.notebook.add(self.tab_recon, text="6. STL Reconstruction")
        
        # เรียกใช้งานฟังก์ชันย่อย ให้เริ่มแปะองค์ประกอบปุ่ม/ตัวหนังสือแยกไปในแต่ละหน้าแท็บ
        self.setup_scan_tab()
        self.setup_processing_tab()
        self.setup_merge_tab()
        self.setup_360_meshing_tab()
        self.setup_turntable_tab()
        self.setup_stl_tab()

    # ==========================================
    # ส่วนของฟังก์ชันจัดสรรหน้าตา GUI ของแต่ละแท็บ
    # ==========================================
    def setup_scan_tab(self):
        # หน้าจอหลักสำหรับ สแกน Calibrate ดึงจุด Point Cloud
        root = self.tab_scan
        
        # ป้ายหัวข้อใหญ่ด้านบนประจำหน้าจอ
        ttk.Label(root, text="3D Scanner Workflow", font=("Arial", 16, "bold")).pack(pady=10)
        # ป้ายบอกเลข IP (เบื้องต้นให้ขึ้น Connecting... ก่อน)
        self.ip_lbl = ttk.Label(root, text="Connecting...", foreground="blue")
        self.ip_lbl.pack()
        # ดึงเลขวงแลนมากางให้มือถือมาร่วมวง
        self.update_ip()
        
        # --- กรอบ STEP 1: Calibrate Capture ---
        lf1 = ttk.LabelFrame(root, text="1. Calibration Capture")
        lf1.pack(fill=tk.X, padx=10, pady=5)
        
        f1_top = ttk.Frame(lf1)
        f1_top.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f1_top, text="Number of Poses:").pack(side=tk.LEFT)
        # ช่องสปินเนอร์ให้กดลูกศรขึ้นลงป้อนจำนวนท่าทาง (ล๊อกระหว่าง 3-20 ท่า)
        ttk.Spinbox(f1_top, from_=3, to=20, textvariable=self.num_poses, width=5).pack(side=tk.LEFT, padx=5)
        
        # ปุ่มเริ่มจับภาพ Calibrate ถ่ายรูปกระดานหมากรุก (เรียกฟังก์ชัน)
        ttk.Button(lf1, text="Capture Calib Images", command=self.do_calib_capture).pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(lf1, text="Save Folder:").pack(anchor=tk.W, padx=5)
        # ช่องกรอก/ดูว่าโฟลเดอร์ไหนถูกเลือก (ผูกกับตัวแปร calib_capture_dir)
        ttk.Entry(lf1, textvariable=self.calib_capture_dir).pack(fill=tk.X, padx=5, pady=(0,5))
        
        # --- กรอบ STEP 2: Calib Process ---
        lf2 = ttk.LabelFrame(root, text="2. Calibration Processing")
        lf2.pack(fill=tk.X, padx=10, pady=5)
        
        # ปุ่มกดคำนวนวิเคราะห์มุมกล้องตามโฟลเดอร์รูปภาพในช่องเซฟ
        ttk.Button(lf2, text="Compute Calibration (Select Folder)", command=self.do_calib_compute).pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(lf2, text="Result File (.mat):").pack(anchor=tk.W, padx=5)
        # ช่องกรอก/ดูว่าไฟล์ .mat ไปหมกอยู่ไหน
        ttk.Entry(lf2, textvariable=self.calib_file).pack(fill=tk.X, padx=5, pady=(0,5))
        
        # --- กรอบ STEP 3: Scan Capture ---
        lf3 = ttk.LabelFrame(root, text="3. Scan Capture")
        lf3.pack(fill=tk.X, padx=10, pady=5)
        
        f3 = ttk.Frame(lf3); f3.pack(fill=tk.X)
        ttk.Label(f3, text="Object Name:").pack(side=tk.LEFT, padx=5)
        # ช่องกรอกชื่อวัตถุ เผื่อสแกนหลายชิ้นจะได้ไม่ทับกัน
        ttk.Entry(f3, textvariable=self.scan_name).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # ปุ่มเริ่มฉายแสงลายขวางลายตั้งเพื่อเก็บพิกัด 3D
        ttk.Button(lf3, text="Capture Scan Images", command=self.do_scan_capture).pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(lf3, text="Scan Folder:").pack(anchor=tk.W, padx=5)
        ttk.Entry(lf3, textvariable=self.scan_capture_dir).pack(fill=tk.X, padx=5, pady=(0,5))
        
        # --- กรอบ STEP 4: Cloud Gen ---
        lf4 = ttk.LabelFrame(root, text="4. Point Cloud Generation")
        lf4.pack(fill=tk.X, padx=10, pady=5)
        
        f4_top = ttk.Frame(lf4)
        f4_top.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f4_top, text="Calib File (.mat):").pack(side=tk.LEFT)
        ttk.Entry(lf4, textvariable=self.calib_file).pack(fill=tk.X, padx=5)
        # เผื่อลืมไฟล์เก่าๆ ก็สามารถชี้เป้า เลือกไฟล์เก่ากลับมาทำงานได้
        ttk.Button(lf4, text="Select .mat File", command=self.select_calib_file).pack(fill=tk.X, padx=5, pady=2)
        
        # ปุ่มปิดจ็อบ ถลุงรหัสสร้างก้อนเมฆพิกัด 3D ออกมา
        ttk.Button(lf4, text="Generate .PLY (Select Scan Folder)", command=self.do_cloud_gen).pack(fill=tk.X, padx=5, pady=5)

    def setup_processing_tab(self):
        # หน้าจอที่ 2: งานเคลียร์ขยะ หักล้างพื้นผนัง
        root = self.tab_proc
        ttk.Label(root, text="Step 2: Cleanup & Process (Batch)", font=("Arial", 14, "bold")).pack(pady=10)
        ttk.Label(root, text="Pipeline: Load -> Remove Background -> Remove Outliers -> Save", foreground="blue").pack()

        # กรอบแหล่งอ้างอิงไฟล์ทั้งหมด (โหลดทีเดียวได้หลายไฟล์ Batch process)
        lf_files = ttk.LabelFrame(root, text="Files")
        lf_files.pack(fill=tk.X, padx=10, pady=5)
        
        # โฟลเดอร์ต้นรัน
        f_in = ttk.Frame(lf_files); f_in.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_in, text="Select Input Folder", command=lambda: self.sel_dir(self.proc_input_dir)).pack(side=tk.LEFT)
        ttk.Entry(f_in, textvariable=self.proc_input_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # โฟลเดอร์จุดหมาย
        f_out = ttk.Frame(lf_files); f_out.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_out, text="Select Output Folder", command=lambda: self.sel_dir(self.proc_output_dir)).pack(side=tk.LEFT)
        ttk.Entry(f_out, textvariable=self.proc_output_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # 1. พารามิเตอร์ของ Background Remove
        lf_bg = ttk.LabelFrame(root, text="1. Background Removal (Plane Segmentation)")
        lf_bg.pack(fill=tk.X, padx=10, pady=5)
        
        f_dist = ttk.Frame(lf_bg); f_dist.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_dist, text="Distance Threshold (default 50.0):").pack(side=tk.LEFT)
        ttk.Entry(f_dist, textvariable=self.bg_dist_thresh, width=10).pack(side=tk.LEFT, padx=5)
        
        f_rn = ttk.Frame(lf_bg); f_rn.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_rn, text="RANSAC n (3) & Iterations (1000):").pack(side=tk.LEFT)
        ttk.Entry(f_rn, textvariable=self.bg_ransac_n, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Entry(f_rn, textvariable=self.bg_iterations, width=8).pack(side=tk.LEFT, padx=5)
        
        # คำใบ้ตัวหนังสือ ช่วยผู้ใช้เข้าใจ
        bg_desc = ("Distance Thresh: Max distance a point can be from the wall plane to be considered 'wall'.\n"
                   "RANSAC n: Points sampled per iteration. Iterations: How many times to try fitting the plane.")
        ttk.Label(lf_bg, text=bg_desc, foreground="#555", justify=tk.LEFT, wraplength=550).pack(padx=5, pady=5)

        # 2. แก๊งกำจัดขยะ (Outliers) ละอองผง
        lf_out = ttk.LabelFrame(root, text="2. Statistical Outlier Removal")
        lf_out.pack(fill=tk.X, padx=10, pady=5)
        
        f_nb = ttk.Frame(lf_out); f_nb.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_nb, text="nb_neighbors (20):").pack(side=tk.LEFT)
        ttk.Entry(f_nb, textvariable=self.proc_nb_neighbors, width=10).pack(side=tk.LEFT, padx=5)
        
        f_std = ttk.Frame(lf_out); f_std.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_std, text="std_ratio (2.0):").pack(side=tk.LEFT)
        ttk.Entry(f_std, textvariable=self.proc_std_ratio, width=10).pack(side=tk.LEFT, padx=5)
        
        out_desc = ("nb_neighbors: Points to analyze around each point. Higher = smoother/safer but slower.\n"
                    "std_ratio: Threshold. Lower (0.5-1.0) = Aggressive removal. Higher (2.0+) = Conservative.")
        ttk.Label(lf_out, text=out_desc, foreground="#555", justify=tk.LEFT, wraplength=550).pack(padx=5, pady=5)

        # ปุ่มเริ่มถลุงงาน Batch processing รวดเดียวจบ
        ttk.Button(root, text="Run Processing Pipeline", command=self.do_batch_processing).pack(fill=tk.X, padx=20, pady=20)
    
    def setup_merge_tab(self):
        # หน้าจอที่ 3: จับโมเดลตะล่อมให้เข้ามุมแล้วปั้นก้อนเดียวกัน
        root = self.tab_merge
        ttk.Label(root, text="Step 3: 360 Degree Merge (Multi-view Alignment)", font=("Arial", 14, "bold")).pack(pady=10)
        
        lf_files = ttk.LabelFrame(root, text="Files")
        lf_files.pack(fill=tk.X, padx=10, pady=5)
        
        # เอาไฟล์สแกนเศษทั้งหมด (ในโฟลเดอร์เดียวกัน) โยนมา
        f_in = ttk.Frame(lf_files); f_in.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_in, text="Select Input Folder (All PLYs)", command=lambda: self.sel_dir(self.merge_input_dir)).pack(side=tk.LEFT)
        ttk.Entry(f_in, textvariable=self.merge_input_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # ชื่อจุดหมายปลายทางของไฟลืที่รวบแผงแล้ว
        f_out = ttk.Frame(lf_files); f_out.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_out, text="Select Output File (.ply)", command=lambda: self.sel_file_save(self.merge_output_file, "PLY")).pack(side=tk.LEFT)
        ttk.Entry(f_out, textvariable=self.merge_output_file).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        lf_param = ttk.LabelFrame(root, text="Parameters")
        lf_param.pack(fill=tk.X, padx=10, pady=5)
        
        f_vx = ttk.Frame(lf_param); f_vx.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_vx, text="Voxel Size (m) [Default 0.02]:").pack(side=tk.LEFT)
        ttk.Entry(f_vx, textvariable=self.merge_voxel, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_vx, text="(Size of downsampling grid. Smaller = distincter but slower/noisier. Larger = coarse alignment.)", foreground="#555").pack(side=tk.LEFT)

        ttk.Button(root, text="Merge 360 Point Clouds", command=self.do_merge_360).pack(fill=tk.X, padx=20, pady=20)

    def setup_360_meshing_tab(self):
        # หน้าจอที่ 4: การถัก Mesh เคลือบพื้นผิวให้กับผลงานผูกขาด 360 องศาโดยเฉพาะ
        root = self.tab_mesh360
        ttk.Label(root, text="Step 4: 360 Meshing (Poisson + Normal Re-orientation)", font=("Arial", 14, "bold")).pack(pady=10)
        
        lf_files = ttk.LabelFrame(root, text="Files")
        lf_files.pack(fill=tk.X, padx=10, pady=5)
        
        f_in = ttk.Frame(lf_files); f_in.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_in, text="Select Input .PLY", command=lambda: self.sel_file_load(self.m360_input_ply, "PLY")).pack(side=tk.LEFT)
        ttk.Entry(f_in, textvariable=self.m360_input_ply).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        f_out = ttk.Frame(lf_files); f_out.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_out, text="Select Output .STL", command=lambda: self.sel_file_save(self.m360_output_stl, "STL")).pack(side=tk.LEFT)
        ttk.Entry(f_out, textvariable=self.m360_output_stl).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        lf_param = ttk.LabelFrame(root, text="Parameters")
        lf_param.pack(fill=tk.X, padx=10, pady=5)
        
        # เลือกทิศทางการถักโครง (Radial, Tangent)
        f_m = ttk.Frame(lf_param); f_m.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_m, text="Orientation Mode:").pack(side=tk.LEFT)
        # Dropdown Combobox ให้คนเลือก
        ttk.Combobox(f_m, textvariable=self.m360_mode, values=["radial", "tangent"], state="readonly", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_m, text="(Radial = Outwards from center | Tangent = Graph consistency)", foreground="#555").pack(side=tk.LEFT)

        f_d = ttk.Frame(lf_param); f_d.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_d, text="Poisson Depth (Default 10):").pack(side=tk.LEFT)
        ttk.Entry(f_d, textvariable=self.m360_depth, width=10).pack(side=tk.LEFT, padx=5)
        
        f_t = ttk.Frame(lf_param); f_t.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_t, text="Density Trim (0.0 = Watertight):").pack(side=tk.LEFT)
        ttk.Entry(f_t, textvariable=self.m360_trim, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_t, text="(0.0 fills EVERYTHING. >0.0 cuts bubbles)", foreground="#555").pack(side=tk.LEFT)
        
        ttk.Button(root, text="Run 360 Meshing", command=self.do_360_meshing).pack(fill=tk.X, padx=20, pady=20)

    def setup_turntable_tab(self):
        # หน้าจอที่ 5 ควบคุมมอเตอร์ Arduino อัตโนมัติ (Turntable)
        root = self.tab_turntable
        ttk.Label(root, text="Step 5: Auto-Scan with Turntable (Arduino)", font=("Arial", 14, "bold")).pack(pady=10)
        
        # 1. กล่องกรอกพอร์ต
        lf_conn = ttk.LabelFrame(root, text="1. Arduino Connection")
        lf_conn.pack(fill=tk.X, padx=10, pady=5)
        
        f_p = ttk.Frame(lf_conn); f_p.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_p, text="Port:").pack(side=tk.LEFT)
        self.cb_port = ttk.Combobox(f_p, textvariable=self.tt_port, width=15)
        self.cb_port.pack(side=tk.LEFT, padx=5)
        ttk.Button(f_p, text="Refresh", command=self.refresh_ports).pack(side=tk.LEFT, padx=2)
        ttk.Button(f_p, text="Connect", command=self.connect_arduino).pack(side=tk.LEFT, padx=5)
        
        # 2. ตั้งค่าระยะหมุน
        lf_set = ttk.LabelFrame(root, text="2. Scan Settings")
        lf_set.pack(fill=tk.X, padx=10, pady=5)
        
        f_deg = ttk.Frame(lf_set); f_deg.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_deg, text="Degrees per Turn (e.g., 30):").pack(side=tk.LEFT)
        ttk.Entry(f_deg, textvariable=self.tt_degrees, width=10).pack(side=tk.LEFT, padx=5)
        
        f_cnt = ttk.Frame(lf_set); f_cnt.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_cnt, text="Number of Turns (e.g., 12):").pack(side=tk.LEFT)
        ttk.Entry(f_cnt, textvariable=self.tt_turns, width=10).pack(side=tk.LEFT, padx=5)
        
        # อัปเดตตัวเลขแสดงผลรวมทุกครั้งที่พิมพ์เลข (เช่น 30 x 12 = 360 รอดตัว!)
        self.lbl_total = ttk.Label(lf_set, text="Total: 360 degrees", foreground="blue")
        self.lbl_total.pack(padx=5, pady=5)
        self.tt_degrees.trace_add("write", self.update_tt_totals)
        self.tt_turns.trace_add("write", self.update_tt_totals)
        
        # 3. กล่องคุมปลายทางเซฟ 
        lf_out = ttk.LabelFrame(root, text="3. Output")
        lf_out.pack(fill=tk.X, padx=10, pady=5)
        
        f_name = ttk.Frame(lf_out); f_name.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_name, text="Base Object Name:").pack(side=tk.LEFT)
        ttk.Entry(f_name, textvariable=self.tt_base_name).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        f_dir = ttk.Frame(lf_out); f_dir.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_dir, text="Select Save Folder", command=lambda: self.sel_dir(self.tt_save_dir)).pack(side=tk.LEFT)
        ttk.Entry(f_dir, textvariable=self.tt_save_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 4. ปุ่มเริ่มรันจักรกลนรก
        ttk.Label(root, textvariable=self.tt_status, font=("Arial", 12)).pack(pady=10)
        ttk.Button(root, text="START AUTO SCAN", command=self.do_auto_scan_sequence, state="normal").pack(fill=tk.X, padx=20, pady=10)

    def setup_stl_tab(self):
        # หน้าจอที่ 6 (สุดท้าย): ขึ้นโครง 3D ปกติสำหรับพ้อยคลาย์แบนๆ 
        root = self.tab_recon
        ttk.Label(root, text="STL Reconstruction", font=("Arial", 14, "bold")).pack(pady=10)
        
        lf_files = ttk.LabelFrame(root, text="Files")
        lf_files.pack(fill=tk.X, padx=10, pady=5)
        
        f_in = ttk.Frame(lf_files); f_in.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_in, text="Select Input .PLY", command=lambda: self.sel_file_load(self.s_input_ply, "PLY")).pack(side=tk.LEFT)
        ttk.Entry(f_in, textvariable=self.s_input_ply).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        f_out = ttk.Frame(lf_files); f_out.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_out, text="Select Output .STL", command=lambda: self.sel_file_save(self.s_output_stl, "STL")).pack(side=tk.LEFT)
        ttk.Entry(f_out, textvariable=self.s_output_stl).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        lf_mode = ttk.LabelFrame(root, text="Method & Parameters")
        lf_mode.pack(fill=tk.X, padx=10, pady=5)
        
        f_m = ttk.Frame(lf_mode); f_m.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_m, text="Reconstruction Mode:").pack(side=tk.LEFT)
        cb = ttk.Combobox(f_m, textvariable=self.s_mode, values=["watertight", "surface"], state="readonly")
        cb.pack(side=tk.LEFT, padx=5)
        # กดเปลี่ยนโหมดแล้วจะซ่อนหน้าต่างพ่วงลูกเล่น
        cb.bind("<<ComboboxSelected>>", self.update_stl_params)
        
        # เฟรมสำหรับโชว์ตัวแปรที่ซ่อนอยู่ (ซ่อน/โผล่ เมื่อโหมด Combox เปลี่ยนค่า)
        self.f_stl_params = ttk.Frame(lf_mode)
        self.f_stl_params.pack(fill=tk.X, padx=5, pady=5)
        self.update_stl_params() # รีเฟซก่อนหนึงครั้ง
        
        ttk.Button(root, text="Run STL Reconstruction", command=self.do_stl_recon).pack(fill=tk.X, padx=20, pady=20)


    # ==========================================
    # ส่วนของฟังก์ชันคำสั่งกดปุ่ม (Actions and Helper Actions)
    # ==========================================

    def update_stl_params(self, event=None):
        # ฟังก์ชันสลับเมนูลูกเล่นในหน้า 6 ตามโหมด (Watertight/Surface)
        for widget in self.f_stl_params.winfo_children():
            widget.destroy() # ลบของเก่าออกไปให้หมดก่อน
            
        mode = self.s_mode.get()
        if mode == "watertight":
            # ถ้าเป็นโหมดตัน ก็จะมีฟิลด์ใส่ค่า Depth
            ttk.Label(self.f_stl_params, text="Poisson Depth (default 10):").pack(anchor=tk.W)
            ttk.Entry(self.f_stl_params, textvariable=self.s_depth).pack(fill=tk.X)
            ttk.Label(self.f_stl_params, text="Creates a closed (watertight) mesh. Higher depth = more detail but slower.", foreground="#555").pack(anchor=tk.W)
        else:
            # ถ้าเป็นพื้นผิวโปรง จะมีแค่ Ball Radii
            ttk.Label(self.f_stl_params, text="Ball Radii Multipliers (default '1, 2, 4'):").pack(anchor=tk.W)
            ttk.Entry(self.f_stl_params, textvariable=self.s_radii).pack(fill=tk.X)
            ttk.Label(self.f_stl_params, text="Multiples of average point distance. Connects dots without filling large holes.", foreground="#555").pack(anchor=tk.W)

    def sel_file_load(self, var, ftype):
        # ฟังก์ชันรองรับการเปิดหน้าต่างให้คนคลิกเลือกไฟล์โหลดเข้ามา (หน้าตากล่อง Dialog ทั่วไป)
        ext = "*.ply" if ftype == "PLY" else "*.*"
        f = filedialog.askopenfilename(filetypes=[(ftype, ext)])
        if f: 
            var.set(f)
            # เติมชื่อไฟล์ช่อง Output (ทางออกให้ล่วงหน้าเลย ถ้าเห็นมันว่างๆ ผู้ใช้จะได้ไม่ต้องพิมพ์เองเสียเวลา)
            if ftype == "PLY":
                # สำหรับโหมดหน้า 6 STL
                if var == self.s_input_ply and not self.s_output_stl.get():
                    self.s_output_stl.set(f.replace(".ply", ".stl"))
                # สำหรับโหมดหน้า 360 Mesh (เผื่อ)
                if var == self.m360_input_ply and not self.m360_output_stl.get():
                    self.m360_output_stl.set(f.replace(".ply", ".stl"))

    def sel_file_save(self, var, ftype):
        # ฟังก์ชันเรียกกล่อง Dialog บันทึกไฟล์ Save As
        ext = "*.ply" if ftype == "PLY" else "*.stl"
        f = filedialog.asksaveasfilename(filetypes=[(ftype, ext)], defaultextension=ext.replace("*", ""))
        if f: var.set(f)

    def sel_dir(self, var):
        # ฟังก์ชันเรียกกล่อง Dialog หน้าต่างเลือกโฟลเดอร์ 
        d = filedialog.askdirectory()
        if d: var.set(d)

    def update_ip(self):
        # ฟังก์ชันเจาะหา IP วงแลนตัวเครื่องเราเอง เอาไปแสดงให้มือถือรู้ว่าต้องพิมพ์เชื่อมที่แอดเดรสไหน
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]; s.close() # ขุด IP ขึ้นมา
            self.ip_lbl.config(text=f"Connect Phone to: http://{ip}:5000") # ตบขึ้นหน้าจอ
        except: pass

    def refresh_ports(self):
        # ให้มันดึงชื่อ COM 1 COM 2 ขึ้นมาประดับบารมีที่ Dropdown (แท่นหมุน)
        ports = self.arduino.get_ports()
        self.cb_port['values'] = ports
        if ports: self.cb_port.current(0) # ท้ารีเฟซแล้วยังไงก็ยงโพล่ เลือกเป็นหลักตังรอเลยอันแรกนั่นละ
    
    def connect_arduino(self):
        # รับการลั่นไกปุ่ม Connect Arduino
        p = self.tt_port.get()
        if not p: messagebox.showerror("Error", "Select a port"); return
        
        ok, msg = self.arduino.connect(p) # จั่วหาพอร์ตว่าติดไหม
        if ok: messagebox.showinfo("Connected", "Arduino Connected!")
        else: messagebox.showerror("Error", f"Failed: {msg}")

    def update_tt_totals(self, *args):
        # เมื่อเปลี่ยนค่า Degree หรือ Turns ในช่องพิมพ์ ก็สลับค่ามาคำณวนองศารวมหน้าจอสดๆ เช่น 15*22=...
        try:
            d = self.tt_degrees.get()
            t = self.tt_turns.get()
            total = d * t
            self.lbl_total.config(text=f"Total: {total} degrees ({t} scans)")
        except: pass

    # --- ฟังก์ชันกระทำการรันประมวลผล (ส่วนสั่งงาน Threading ทำคู่ขนานเพื่อไม่ให้หน้าจอค้าง) ---

    def do_calib_capture(self):
        # รับปุ่มสั่งการทำงานขั้นแรก: ถ่ายเก็บภาพ Calibration 
        d = self.calib_capture_dir.get()
        n = self.num_poses.get()
        # เริ่มการสาดแสงกากบาดใส่หน้าต่างมื้อมือ ผ่านด้ายคู่ขนาน (Thread) ยอมให้โปรแกรมอื่นในคอมทำต่อ
        threading.Thread(target=self.sys.capture_calibration, args=(d, n), daemon=True).start()

    def do_calib_compute(self):
        # ขั้นคำนวณ Calibrate ย่อย
        initial = self.calib_capture_dir.get()
        if not os.path.exists(initial): initial = os.getcwd()
        
        in_dir = filedialog.askdirectory(title="Select Calibration Images Folder", initialdir=initial)
        if not in_dir: return
        
        self.calib_capture_dir.set(in_dir)
        out_file = os.path.join(in_dir, "calib.mat")
        
        threading.Thread(target=self.run_calib_analysis, args=(in_dir, out_file), daemon=True).start()

    def run_calib_analysis(self, in_dir, out_file):
        try:
            # ดึงวิเคราะห์ Errors คืนค่า
            errors, available_poses = self.sys.analyze_calibration(in_dir)
            # เด้งหน้าต่างให้คนตัดสินใจบน Thread หน้าจอหลัก (main thread)
            self.root.after(0, self.prompt_pose_selection, errors, available_poses, in_dir, out_file)
        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Calib Error", err_msg))

    def prompt_pose_selection(self, errors, available_poses, in_dir, out_file):
        # โชว์รั้งท้ายความคลาดเคลื่อน ถามว่าจะตัดรูปไหนทิ้งไหม 
        msg = "Calibration Analysis (Error in px):\n\n"
        for pose, (ce, pe) in errors.items():
            msg += f"{pose}: Cam={ce:.2f}, Proj={pe:.2f}\n"
        msg += "\nEnter poses to KEEP (e.g., '1,3,4' OR 'all' for all):"
        
        user_input = simpledialog.askstring("Select Poses", msg, parent=self.root)
        if not user_input: return
        
        selected_poses = []
        user_input = user_input.strip()
        
        if user_input.lower() == 'all':
            selected_poses = available_poses
        else:
            selected_indices = [x.strip() for x in user_input.split(',')]
            for idx in selected_indices:
                name = f"pose_{idx}"
                if idx.startswith("pose_"): name = idx
                if name in available_poses: selected_poses.append(name)
        
        # ปล่อยผีเดินหน้า Calibrate ตัวเมียต่อไป
        threading.Thread(target=self.run_calib_final, args=(in_dir, selected_poses, out_file), daemon=True).start()

    def run_calib_final(self, in_dir, selected_poses, out_file):
        try:
            self.sys.calibrate_final(in_dir, selected_poses, out_file)
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Calibration Saved to:\n{out_file}"))
            self.root.after(0, lambda: self.calib_file.set(out_file)) # เซ็ตไฟล์ที่ได้ลงช่องให้เลย
        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Calib Final Error", err_msg))

    def do_scan_capture(self):
        # สั่งกระโดดถ่ายภาพ Scan ถอดรหัสลายขวางลายตั้ง
        base = os.path.join(DEFAULT_ROOT, "scans")
        name = self.scan_name.get()
        path = os.path.join(base, name)
        self.scan_capture_dir.set(path)
        
        threading.Thread(target=self.sys.capture_scan, args=(path,), daemon=True).start()

    def select_calib_file(self):
        initial = self.calib_file.get()
        if not initial or not os.path.exists(os.path.dirname(initial)): initial = os.getcwd()
        f = filedialog.askopenfilename(title="Select Calibration .mat", initialdir=os.path.dirname(initial), filetypes=[("MAT Files", "*.mat")])
        if f: self.calib_file.set(f)

    def do_cloud_gen(self):
        # รันสร้างเมฆ Point Cloud
        initial = self.scan_capture_dir.get()
        if not os.path.exists(initial): initial = os.getcwd()
        
        scan_dir = filedialog.askdirectory(title="Select Scan Images Folder", initialdir=initial)
        if not scan_dir: return
        
        calib_path = self.calib_file.get()
        
        threading.Thread(target=self.run_cloud_gen, args=(scan_dir, calib_path), daemon=True).start()

    def run_cloud_gen(self, scan_dir, calib_path):
        try:
            self.sys.generate_cloud(scan_dir, calib_path)
            self.root.after(0, lambda: messagebox.showinfo("Done", f"Cloud generation finished for\n{os.path.basename(scan_dir)}"))
        except Exception as e:
             err_msg = str(e)
             self.root.after(0, lambda: messagebox.showerror("Error", err_msg))

    def do_batch_processing(self):
        # รันหน้าสอง (Batch ลบทิ้งขยะ + พื้นหลัง)
        in_dir = self.proc_input_dir.get()
        out_dir = self.proc_output_dir.get()
        
        # BG Params
        bg_dist = self.bg_dist_thresh.get()
        bg_rn = self.bg_ransac_n.get()
        bg_iters = self.bg_iterations.get()
        
        # Outlier Params
        nb = self.proc_nb_neighbors.get()
        std = self.proc_std_ratio.get()
        
        if not in_dir or not out_dir:
            messagebox.showerror("Error", "Select both Input and Output folders.")
            return

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        ply_files = glob.glob(os.path.join(in_dir, "*.ply"))
        if not ply_files:
            messagebox.showerror("Error", "No .ply files found in input folder.")
            return

        def run():
            count = 0
            errors = 0
            total = len(ply_files)
            
            for fpath in ply_files:
                fname = os.path.basename(fpath)
                out_path = os.path.join(out_dir, fname.replace(".ply", "_processed.ply"))
                
                print(f"[Task] Processing {fname}...")
                try:
                    pcd = self.processor.remove_background(fpath, distance_threshold=bg_dist, ransac_n=bg_rn, num_iterations=bg_iters, return_obj=True)
                    pcd = self.processor.remove_outliers(pcd, nb_neighbors=nb, std_ratio=std, return_obj=True)
                    
                    import open3d as o3d
                    o3d.io.write_point_cloud(out_path, pcd)
                    print(f"[Task] Saved {out_path}")
                    count += 1
                except Exception as e:
                    print(f"[Task] Error processing {fname}: {e}")
                    errors += 1
            
            msg = f"Pipeline Completed.\nProcessed: {count}/{total}\nErrors: {errors}\nSaved to: {out_dir}"
            self.root.after(0, lambda: messagebox.showinfo("Processing Done", msg))
        
        threading.Thread(target=run, daemon=True).start()

    def do_merge_360(self):
        # รันหน้าสาม ต่อโมเดล 360 ดริกิ๊
        in_dir = self.merge_input_dir.get()
        out_file = self.merge_output_file.get()
        vx = self.merge_voxel.get()
        
        if not in_dir or not out_file:
            messagebox.showerror("Error", "Select Input Folder and Output File.")
            return

        def run():
            try:
                self.processor.merge_pro_360(in_dir, out_file, vx)
                self.root.after(0, lambda: messagebox.showinfo("Merge Done", f"Saved merged cloud to:\n{out_file}"))
            except Exception as e:
                 err_msg = str(e)
                 print(err_msg)
                 self.root.after(0, lambda: messagebox.showerror("Error", err_msg))
        
        threading.Thread(target=run, daemon=True).start()

    def do_360_meshing(self):
        # รันหน้า 4 โบท็อกซ์ Mesh ปกติ 
        i = self.m360_input_ply.get()
        o = self.m360_output_stl.get()
        d = self.m360_depth.get()
        t = self.m360_trim.get()
        m = self.m360_mode.get()
        
        if not i or not o: messagebox.showerror("Error", "Select files first."); return
        
        def run():
            try:
                self.processor.mesh_360(i, o, d, t, m)
                self.root.after(0, lambda: messagebox.showinfo("Done", f"360 Mesh Saved to:\n{o}"))
            except Exception as e:
                err_msg = str(e)
                print(f"Error: {e}")
                # Don't popup error if thread dying... print is safer.
        
        threading.Thread(target=run, daemon=True).start()

    def do_stl_recon(self):
        # รันหน้า 6 ประชันงานธรรมดาไม่ต่อพ่วง
        i = self.s_input_ply.get()
        o = self.s_output_stl.get()
        m = self.s_mode.get()
        
        params = {}
        if m == "watertight": params["depth"] = self.s_depth.get()
        else: params["radii"] = self.s_radii.get()
        
        if not i or not o: messagebox.showerror("Error", "Select files first."); return
        
        def run():
            try:
                self.processor.reconstruct_stl(i, o, m, params)
                self.root.after(0, lambda: messagebox.showinfo("Done", f"STL Saved to:\n{o}"))
            except Exception as e:
                err_msg = str(e)
                self.root.after(0, lambda: messagebox.showerror("Error", err_msg))
        
        threading.Thread(target=run, daemon=True).start()

    def do_auto_scan_sequence(self):
        # รันหน้า 5 (พระเอก) เปิดศึกมุดรอบก้านหมุน Turntable ออโต้สแกน 
        
        # ตรวจสอบการพึ่งพา Arduino
        if not self.arduino.ser:
            # ถ้าไม่มีกลมืน ถามว่าตะลุยต่อลักไก่เล่นๆ (Simulation) ไหม
            if not messagebox.askyesno("Confirm", "Arduino not connected (in software). Continue anyway (Simulation)?"):
                return
        
        deg = self.tt_degrees.get()
        turns = self.tt_turns.get()
        base_name = self.tt_base_name.get()
        root_dir = self.tt_save_dir.get()
        
        if not base_name or not root_dir:
            messagebox.showerror("Error", "Check Output settings"); return
            
        # Create Main Folder (โฟลเดอร์รันชิ้นงาน 360 สำหรับหอบใหญ่นี้)
        main_folder = os.path.join(root_dir, f"{base_name}_{int(deg)}deg_AUTO")
        os.makedirs(main_folder, exist_ok=True)
        
        # ฟลอร์เต็นท์ใหม่ Popup Progress (หน้าต่างรอง แจ้งคืวความก้าวหน้าระหว่างรัน)
        top = tk.Toplevel(self.root)
        top.title("Auto Scan Progress")
        top.geometry("400x300")
        
        lbl_info = ttk.Label(top, text="Starting...", font=("Arial", 12))
        lbl_info.pack(pady=20)
        
        lbl_time = ttk.Label(top, text="Time: 0s")
        lbl_time.pack(pady=5)
        
        pb = ttk.Progressbar(top, maximum=turns, mode='determinate')
        pb.pack(fill=tk.X, padx=20, pady=20)
        
        # Thread Logic ขบวนการออโต้ไล่ฟัน
        def run_thread():
            start_time = time.time()
            
            for i in range(turns): # วิ่งรอบกี่ปืนโตตามรอบ (turns)
                # Update UI (อัปเดตสภาพบนหน้าจอ UI ที่มีโชว์)
                elapsed = time.time() - start_time
                avg_time = (elapsed / i) if i > 0 else 0
                rem_time = avg_time * (turns - i)
                
                msg = f"Scanning {i+1}/{turns}\nElapsed: {int(elapsed)}s\nEst. Left: {int(rem_time)}s"
                
                self.root.after(0, lambda: lbl_info.config(text=msg))
                self.root.after(0, lambda: lbl_time.config(text=f"Time: {int(elapsed)}s"))
                self.root.after(0, lambda m=i: pb.config(value=m))
                
                # 1. CAPTURE ถ่ายรูปรัวๆ
                current_angle = i * deg
                sub_name = f"{base_name}_{int(current_angle)}deg_scan" # ชื่อย่อยท่าทาง
                sub_path = os.path.join(main_folder, sub_name)
                
                print(f"[Auto] Capturing to {sub_path}")
                
                try:
                    # ป้อนคำสั่งลับ silent=True คือระเว้นการจ้อหน้าจอโชว์เตือนใดๆทั้งปวง ระหว่างสาดแสง ให้มันไม่ติดขัด
                    self.sys.capture_scan(sub_path, silent=True)
                except Exception as e:
                    print(f"Scan Error: {e}")
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Scan failed: {e}"))
                    return

                # 2. MOVE หมุนแท่นเต้นรำไปเตรียมรับดาบต่อไป
                if i < turns - 1: # ถ้ายังไม่ถึงรอบสุดท้ายก็สั่งมอเตอร์วิ่ง
                    msg_move = f"Rotating {deg} degrees..."
                    self.root.after(0, lambda: lbl_info.config(text=msg_move))
                    
                    if self.arduino.ser:
                        self.arduino.rotate(deg)
                        # เฝ้ารอคำว่า DONE จาก Arduino โดยให้เวทีเวลา 10 วิ ถ้าโต้ตอบไม่ทันแสดงว่าแท่นอาจฝืดหรือไม่รัน
                        done = self.arduino.wait_for_done(timeout=10) 
                        if not done:
                            print("Warning: Arduino move timeout or no DONE received.")
                        time.sleep(0.5) # พักหน่วงนึดนึงกันการสั่นสะเทือนชิ้นงาน
                    else:
                        time.sleep(2) # ซิมูลล่าเล่นๆไปพลางๆตอนเทสคอม

            # Finish (รวบตึง)
            total_time = time.time() - start_time
            done_msg = f"Auto Scan Complete!\nTotal Time: {int(total_time)}s\nLocation: {main_folder}"
            self.root.after(0, lambda: messagebox.showinfo("Done", done_msg))
            self.root.after(0, top.destroy) # ยกทัพปิดหน้าจอ ProgressBar

        threading.Thread(target=run_thread, daemon=True).start()

