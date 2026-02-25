import os
import cv2
import glob
import time
import uuid
import numpy as np
import scipy.io
from tkinter import messagebox

# นำเข้าตัวแปรคอนฟิก (Configuration) และสถานะเซิร์ฟเวอร์ จากไฟล์อื่น
from config import SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_OFFSET_X, PROJ_VALUE, D_SAMPLE_PROJ, CHECKER_ROWS, CHECKER_COLS, SQUARE_SIZE
from server import SERVER_STATE

class SLSystem:
    # คลาสสำหรับจัดการระบบสแกนด้วยแสงโครงสร้าง (Structured Light System)
    # รับหน้าที่ตั้งแต่การฉายแสง ควบคุมแพทเทิร์น และวิเคราะห์หาระยะ 3 มิติ
    def __init__(self):
        # ---------------------------------------------------------
        # ชื่อหน้าต่างของโปรเจกเตอร์ที่จะถูกเปิดขึ้นมาขยายเต็มจอ
        self.window_name = "Projector"
        
    def init_projector(self):
        # ---------------------------------------------------------
        # ฟังก์ชันเปิดใช้งานระบบแสดงผลของโปรเจกเตอร์
        # สร้างหน้าต่าง OpenCV (สามารถย่อขยายได้)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # ย้ายหน้าต่างนี้ไปยังตำแหน่งหน้าจอที่ 2 (ตามระยะ OFFSET ที่ตั้งไว้ เช่น 1920)
        cv2.moveWindow(self.window_name, SCREEN_OFFSET_X, 0)
        # บังคับหน้าต่างโปรเจกเตอร์ให้กินพื้นที่เต็มขอบจอ (Fullscreen) ตัดแถบเมนูออก
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # สร้างภาพพื้นผิวดำมืดสนิท ขนาดเท่าความละเอียดโปรเจกเตอร์
        black = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8)
        # โชว์ภาพสีดำ (เพื่อล้างหน้าจอและเตรียมพร้อม)
        cv2.imshow(self.window_name, black)
        # รอระบบวาดภาพเสร็จ 50 มิลลิวินาที
        cv2.waitKey(50)

    def close_projector(self):
        # ---------------------------------------------------------
        # ฟังก์ชันปิดหน้าต่างโปรเจกเตอร์
        cv2.destroyWindow(self.window_name)

    def generate_patterns(self):
        # ---------------------------------------------------------
        # ฟังก์ชันสร้างชุดลวดลายแถบแสงขาว-ดำ (Gray Code) สำหรับฉายไปบนวัตถุ
        
        # คำนวณความละเอียดเอาต์พุต (ถ้ามีการ Downsample ขนาดภาพจะเล็กลง)
        width, height = SCREEN_WIDTH // D_SAMPLE_PROJ, SCREEN_HEIGHT // D_SAMPLE_PROJ
        
        # คำนวณจำนวนภาพบิต (จำนวนครั้งการแบ่งครึ่ง) ที่จำเป็นสำหรับแกน X (แนวตั้ง)
        n_cols = int(np.ceil(np.log2(width)))
        # คำนวณจำนวนภาพบิต ที่จำเป็นสำหรับแกน Y (แนวนอน)
        n_rows = int(np.ceil(np.log2(height)))
        
        def get_gray_1d(n):
            # ฟังก์ชันอัลกอริทึมในการเรียงตัวเลขฐานสองให้เป็นแบบ Gray Code (เปลี่ยนบิตเพียง 1 ตำแหน่งต่อครั้ง)
            # ตัวอย่างเช่น 00, 01, 11, 10
            if n == 1: return ['0', '1'] # ถ้าต้องการบิตเดียว (ระดับแรก)
            prev = get_gray_1d(n - 1)  # สั่งให้รันฟังก์ชันตัวเองซ้ำๆ เพื่อสร้างฐานก่อนหน้า
            # นำฐานเดิมเติมเลข 0 ด้านหน้า ควบรวมกับฐานเดิมที่ดึงถอยหลังเติม 1 ลงไป
            return ['0' + s for s in prev] + ['1' + s for s in prev[::-1]]

        # สร้างแบบแผนความถี่ Gray Code สำหรับแนวตั้งและแนวนอน
        col_gray = get_gray_1d(n_cols)
        row_gray = get_gray_1d(n_rows)
        
        P = [[], []] # เตรียมลิสต์ 2 มิติ (ช่อง [0] สำหรับแนวตั้ง, [1] สำหรับแนวนอน)
        
        # สร้างภาพลวดลายแนวตั้งทีละบิตภาพ
        for b in range(n_cols):
            pat = np.zeros((height, width), dtype=np.uint8) # พื้นดำ
            for c in range(width): # วนตามความกว้าง
                # ถ้ารหัสคอลัมน์นั้นๆ บิตนี้เป็น '1' ให้เติมสีขาวลงไปทั้งเส้นคอลัมน์นั้น
                if c < len(col_gray) and col_gray[c][b] == '1': pat[:, c] = 1
            P[0].append(pat) # เก็บเข้าลิสต์แนวตั้ง

        # สร้างภาพลวดลายแนวนอนทีละบิตภาพ
        for b in range(n_rows):
            pat = np.zeros((height, width), dtype=np.uint8) # พื้นดำ
            for r in range(height): # วนตามความสูง
                # ถ้ารหัสแถวนั้นๆ บิตนี้เป็น '1' ให้เส้นขวางเป็นแถบสว่าง
                if r < len(row_gray) and row_gray[r][b] == '1': pat[r, :] = 1
            P[1].append(pat) # เก็บเข้าลิสต์แนวนอน
            
        return P # ส่งคืนชุดลวดลายทั้งหมดออกไป

    def trigger_capture(self, save_path):
        # ---------------------------------------------------------
        # ฟังก์ชันสั่งการถ่ายภาพผ่านเซิร์ฟเวอร์ และรอรับรูป
        
        # ล้างสถานะ Event รอรูปเดิมออกไปก่อน (ปรับเป็นสถานะยังไม่ได้รับรูป)
        SERVER_STATE["upload_received_event"].clear()
        
        # บอกระบบหลังบ้านให้รู้ว่าเดี๋ยวรูปถัดไปต้องเซฟทับหรือเก็บไว้ที่ชื่อนี้
        SERVER_STATE["last_image_path"] = save_path
        # สร้างใบสั่งคิวที่ไม่ซ้ำกัน
        SERVER_STATE["command_id"] = str(uuid.uuid4())
        # ระบุว่าตอนนี้เซิร์ฟเวอร์ต้องปล่อยคำสั่ง "capture" ให้มือถือแล้วนะ
        SERVER_STATE["command"] = "capture"
        
        # ตัวโปรแกรมเราจะหยุดค้างรอ (Wait) สัญญาณว่า "ได้รับรูปแล้ว" อัปเดตจากฟังก์ชันรับอัปโหลด (มากสุดรอ 20 วินาที)
        if not SERVER_STATE["upload_received_event"].wait(timeout=20):
            print(f"[Error] Timeout capturing {save_path}") # ถ้าเกิดนานเกินไป ถือว่าพัง
            return False
            
        # เมื่อได้รูปแล้ว กลับสู่วิถีปกติ "idle" ว่างงาน
        SERVER_STATE["command"] = "idle"
        return True # แจ้งคนสั่งมาว่าทำสำเร็จ

    # ---------------------------------------------------------
    # 1. CAPTURE CALIBRATION (ขั้นตอนที่ 1: การสะสมภาพ Calibrate กล้องกับโปรเจกเตอร์)
    # ---------------------------------------------------------
    def capture_calibration(self, save_dir, num_poses=5):
        # รับค่าจำนวนท่า (Pose) ที่ต้องการ
        try:
            num_poses = int(num_poses)
        except:
            num_poses = 5
            
        # เปิดหน้าต่างโปรเจกเตอร์
        self.init_projector()
        # คำนวณและดึงลวดลายที่จะฉายมาไว้ในหน่วยความจำ
        P = self.generate_patterns()
        
        # เตรียมที่จัดเก็บภาพพื้นฐานที่ต้องใช้ทุกๆท่าทาง
        patterns = []
        # แสงขาว (เปิดยิงแสงสว่างทั้งหมด) (ให้เห็นกระดานรอบทิศ)
        white = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8) * PROJ_VALUE
        # แสงดำมืด 
        black = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8)
        
        patterns.append(("01.png", white)) # ภาพแรก
        patterns.append(("02.png", black)) # ภาพสอง
        
        idx = 3 # ภาพแถบความกว้างเริ่มที่ใบ 3
        # แผ่ขยายลวดลาย P ที่คราฟมาให้มาเป็นภาพคู่ (แสงปกติสลับกับแสงผกผัน Inverse)
        for j in range(2): 
            for pat in P[j]:
                pat_img = (pat * PROJ_VALUE).astype(np.uint8) # ลวดลายแสงตามรหัส
                inv_img = ((1-pat) * PROJ_VALUE).astype(np.uint8) # ส่วนกลับ (Inverse) ของลวดลายด้านบน
                
                # หากมีการย่อส่วนลวดลายแต่แรก ก็ขยายกลับให้เต็มจอก่อนยิงแสงจริง
                if D_SAMPLE_PROJ > 1:
                    pat_img = cv2.resize(pat_img, (SCREEN_WIDTH, SCREEN_HEIGHT), interpolation=cv2.INTER_NEAREST)
                    inv_img = cv2.resize(inv_img, (SCREEN_WIDTH, SCREEN_HEIGHT), interpolation=cv2.INTER_NEAREST)
                    
                # เก็บลงลิสต์ พร้อมตั้งชื่อไฟล์ที่จะให้ถ่าย
                patterns.append((f"{idx:02d}.png", pat_img)); idx+=1
                patterns.append((f"{idx:02d}.png", inv_img)); idx+=1

        # สร้างโฟลเดอร์สำหรับเก็บภาพเซ็ตใหญ่นี้
        os.makedirs(save_dir, exist_ok=True)
        # เด้งหน้าต่างแจ้งเตือนผู้ใช้ให้รับทราบ ว่าเริ่มตระเตรียมการกล้อง
        messagebox.showinfo("Step 1", f"Starting Calibration Capture ({num_poses} poses).\nImages will be saved to:\n{save_dir}")
        
        # วนรอบถ่ายและฉายแสงทีละท่าทาง (Pose 1, Pose 2, ...)
        for pose in range(1, num_poses + 1):
            pose_dir = os.path.join(save_dir, f"pose_{pose}")
            os.makedirs(pose_dir, exist_ok=True) # โฟลเดอร์สำหรับท่านี้
            
            # เปิดจอโปรดให้ขาวสว่างสุด เพื่อให้คนจัดกระดานเห็นได้ชัดเจน
            cv2.imshow(self.window_name, white)
            cv2.waitKey(50)
            
            # แจ้งผู้ใช้ว่า "รอบที่ x แล้ว ขยับกระดานหมากรุกให้มุมมันเปลี่ยน แล้วกด OK ดำเนินการ"
            messagebox.showinfo("Calibration", f"Pose {pose}/{num_poses}.\nMove board then click OK.")
            
            # เมื่อผู้ใช้กด OK แล้ว จึงเริ่มวนลูปฉายลวดลายทีละรูปๆ (ไล่จาก 01 ถึงใบสุดท้ายตามลิสต์) 
            for fname, img in patterns:
                cv2.imshow(self.window_name, img) # ฉายลวดลายขึ้นโปรเจกเตอร์
                cv2.waitKey(250) # หน่วงเวลาให้แสงจากหน้าจอเข้าเซ็นเซอร์กล้องเต็มที่
                
                # สั่งกดชัตเตอร์และบันทึก
                if not self.trigger_capture(os.path.join(pose_dir, fname)):
                    messagebox.showerror("Error", "Capture timeout.") # ถ้ามือถือแบตหมดหรือเน็ตหลุด แจ้งตาย
                    self.close_projector()
                    return

        # ปิดโปรเจกเตอร์หลังเสร็จสิ้น 100%
        self.close_projector()
        messagebox.showinfo("Step 1 Done", "Calibration Capture Complete.")

    # ---------------------------------------------------------
    # 2. PROCESS CALIBRATION (ขั้นตอนที่ 2: วิเคราะห์หาระยะความเหลื่อมกล้อง)
    # ---------------------------------------------------------
    def analyze_calibration(self, input_dir):
        # ฟังก์ชันวิเคราะห์ท่าและการทับซ้อน เพื่อประเมินความคลาดเคลื่อนเบื้องต้นว่ากระดานจับมุมได้ครบไหม
        
        # ค้นหาโฟลเดอร์ท่าทางต่างๆ (Poses) ทั้งหมดในโฟลเดอร์เก็บรูป
        available_poses = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
        
        # ต้องมีอย่างน้อย 3 ท่าทางเพื่อให้สมการสเตอริโอสามมิติมีความแม่นยำ (อ้างอิงจากหลักรูปสามเหลี่ยมเรขาคณิต)
        if len(available_poses) < 3: 
            raise ValueError(f"Need at least 3 pose folders in {input_dir}")

        print(f"[Calib] analyzing {len(available_poses)} poses...")
        
        # วิเคราะห์ค่าความคลาดเคลื่อนผลลัพธ์ (Reprojection Error) ของแต่ละท่าทางที่ถ่ายไว้ 
        errors = self.compute_reprojection_errors(input_dir, available_poses)
        
        return errors, available_poses # ส่งรายการท่าและระดับความเพี้ยนกลับไปให้ผู้ใช้คัดกรองต่อ

    def load_calib_data(self, base_dir, pose_list):
        # ฟังก์ชันหัวใจสำคัญ ในการถอดรหัสแสงอ่านพิกัดจากรูปภาพการ Calibrate ชุดใหญ่
        # ถือเป็นการหา "จุดจับคู่" ระหว่าง (จุดบนกล้อง <-> จุดจริงบนโลก <-> จุดหน้าจอโปรเจกเตอร์)
        
        # สร้างพิกัด 3D สมมติฐานแบบเป๊ะๆ ของกระดานหมากรุก (เช่น พิกัดที่ 0: 0,0,0 พิกัดต่อไป 1,0,0...) เป็นมิติโลก 
        objp = np.zeros((CHECKER_ROWS * CHECKER_COLS, 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHECKER_ROWS, 0:CHECKER_COLS].T.reshape(-1, 2)
        objp *= SQUARE_SIZE # คูณด้วยขนาดจริงของช่องสี่เหลี่ยมหมากรุก (เช่น 35 มม.)
        
        # ที่เก็บอาร์เรย์สรุปยอด
        obj_pts = []  # จุดตารางบนโลก (Object Points)
        cam_pts = []  # จุดที่หาเจอบนภาพถ่าย (Camera Points)
        proj_pts = [] # จุดบนจอโปรเจกเตอร์ตามการถอดรหัสแสง (Projector Points)
        valid_poses = [] # เก็บเฉพาะท่าทางที่ภาพสมบูรณ์และใช้งานได้จริงๆ
        img_shape = None # ขนาดภาพกล้อง
        
        # ไล่เปิดประมวลผลไปทีละท่า
        for pose in pose_list:
            path = os.path.join(base_dir, pose)
            
            # เปิดภาพฉายจอขาวสุด (01.png) มาเพื่อหาร่องรอยตารางกระดานหมากรุกง่ายที่สุด
            img = cv2.imread(os.path.join(path, "01.png")) 
            if img is None: continue # ไฟล์พังก็ข้ามมเลย
            if img_shape is None: img_shape = (img.shape[1], img.shape[0]) # สรุปขนาดแชนเบลรูป
            
            # ปรับแต่งภาพขาวให้เด่นชัดขึ้น
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # แปลงเป็นขาวดำ
            blurred = cv2.GaussianBlur(gray, (5, 5), 0) # เบลอกำจัดนอยซ์ (Gaussian Blur)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # ดึงความต่างแสงเงา (Adaptive Histogram Equalization)
            enhanced = clahe.apply(blurred) 
            
            # พยายามหาจุดตัดของกระดานหมากรุกในภาพว่าอยู่ไหน
            ret, corners = cv2.findChessboardCorners(enhanced, (CHECKER_ROWS, CHECKER_COLS), None)
            
            if ret: # ถ้าหาเจอกระดานหมากรุกได้สำเร็จ
                # ปรับแต่งพิกัดรอยต่อที่หามาให้มีความละเอียดเจาะลึกลงระดับ sub-pixel (แม่นยำกว่า 1 พิกเซลเล็กๆซะอีก)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                
                # เริ่มสกัดแสงโครงสร้าง (Decode Structured Data)
                files = sorted(glob.glob(os.path.join(path, "*.png")))
                
                # เช็คขีดสุดกี่บิตจอภาพ
                n_col_bits = int(np.ceil(np.log2(SCREEN_WIDTH)))
                n_row_bits = int(np.ceil(np.log2(SCREEN_HEIGHT)))
                
                # ตรวจสอบว่ามีจำนวนภาพแสงโครงสร้างครบถ้วนหรือไม่ (หักภาพจอดำกะขาวออก)
                if len(files) - 2 < 2 * (n_col_bits + n_row_bits):
                    print(f"Skipping {pose}: Not enough images.")
                    continue
                
                base_idx = 2 # เริ่มอ่านภาพคู่จากลำดับที่ 3 เป็นต้นไป (Index = 2)
                
                # ฟังก์ชันย่อยสำหรับดูดซับลำดับภาพบิต (Binary/Gray Code) มาแปลงเป็นระนาบ X,Y ของจอฉาย
                def decode_seq(n_bits, idx_start): 
                    # ฟังก์ชันนี้ใช้การเทียบภาพปติ (Positive) กับภาพตรงข้ามของมัน (Inverse) วินิจฉัยว่าเป็นบิต 1 หรือ 0 
                    val = np.zeros(len(corners2)) # ลิสต์เก็บข้อมูลสำหรับทุกๆพิกเซลเป้าหมาย มุมกระดานหมากรุก
                    idx = idx_start
                    bin_code = None
                    for b in range(n_bits):  # อ่านคู่รูปลงลึกทีละบิตลึกสุดทลายจอ
                        img_p = cv2.imread(files[idx], 0)     # รูปจริง
                        img_i = cv2.imread(files[idx+1], 0)   # รูป Inverse
                        idx += 2
                        
                        x = corners2[:,0,0] # พิกัดความกว้างซับพิกเซล
                        y = corners2[:,0,1] # พิกัดความสูงซับพิกเซล
                        
                        # ดึงค่าเฉดยิบย่อยจากภาพปกติ และภาพผกผัน
                        vp = img_p[y.astype(int), x.astype(int)]
                        vi = img_i[y.astype(int), x.astype(int)]
                        
                        # แยกความน่าจะเป็น ถ้าส่วนไหนสว่างกว่า Inverse ส่วนนั้นตกเป็นเลขฐาน '1' สำหรับรูปนั่นๆ
                        bit = (vp > vi).astype(int)
                        
                        # พลิกแพลงเข้ารหัสตามกฎ Gray Binary
                        if b == 0: 
                            bin_code = bit
                        else: 
                            # นำส่วนต่างมาเข้าสมการ Exclusive OR หาบิตแท้จริง (XOR operation)
                            bin_code = np.bitwise_xor(bin_code, bit)
                        
                        # คูณทายผลน้ำหนักบิต เลื่อนลงเป็นเลขพิกเซลจอ 1080 (Shift bits + Decimal convert)
                        val += bin_code * (2**(n_bits - 1 - b))
                    
                    # รีดเลขทศนิยมจอ และรหัสคิวต่อรอบหน้า (val = ตำแหน่ง 2D, idx = ตำแหน่งรูปชุดถัดไป)
                    return val, idx 
                
                # เรียกใช้ซะ ทลวงค่าหน้าจอกว้างกับสูงให้มาประสาน ณ องค์ประกอบรูปตารางตัวจริง
                col_val, base_idx = decode_seq(n_col_bits, base_idx) # แกนตั้ง
                row_val, base_idx = decode_seq(n_row_bits, base_idx) # แกนนอน
                
                # column_stack : เอาคู่ชูคอลัมน์กะแถวมาประกอบร่าง (Column 50กะแถว 100 ดองรวมเป็น => [50, 100])
                # reshape(-1, 1, 2): จัดรูปโรมรันให้เข้าข้อกำหนดอัลกอริทึมของ OpenCV 
                proj_pts_pose = np.column_stack((col_val, row_val)).astype(np.float32).reshape(-1, 1, 2)
                
                # ยัดลงกรุพุงสำรองเป็นชุดเดียว 
                obj_pts.append(objp)
                cam_pts.append(corners2)
                proj_pts.append(proj_pts_pose)
                valid_poses.append(pose)
                
        # ส่งค่ากระตุกเชือกคืนแม่
        return obj_pts, cam_pts, proj_pts, img_shape, valid_poses

    def compute_reprojection_errors(self, base_dir, pose_list):
        # ฟังก์ชันคำนวณคาดเดาความคลาดเคลื่อนเบื้องต้นแบบ Quick Calib หาแต้มกะยอดความผิดเพี้ยนของแต้ม (Reprojection)
        obj_pts, cam_pts, proj_pts, shape, poses = self.load_calib_data(base_dir, pose_list)
        
        # ปรับเทียบหน้ากล้องแบบเร็ว 
        rc, mc, dc, rvc, tvc = cv2.calibrateCamera(obj_pts, cam_pts, shape, None, None) 
        # ปรับเทียบกล้องแสงโปรเจกเตอร์แบบเร็ว (โดยอนุมานให้มันเป็นเสมือนกล้องรับภาพที่ซูม)
        rp, mp, dp, rvp, tvp = cv2.calibrateCamera(obj_pts, proj_pts, (SCREEN_WIDTH, SCREEN_HEIGHT), None, None)
        
        errors = {}
        for i, p in enumerate(poses):
            # จำลองสาดเส้นกลับลงเลนส์ แล้ววัดความเบี่ยงเบนว่าจุดจำลองกับจุดถ่ายจริงหลุดโลกไปแค่ไหน
            p2_c, _ = cv2.projectPoints(obj_pts[i], rvc[i], tvc[i], mc, dc)
            err_c = cv2.norm(cam_pts[i], p2_c, cv2.NORM_L2)/len(p2_c)
            
            p2_p, _ = cv2.projectPoints(obj_pts[i], rvp[i], tvp[i], mp, dp)
            err_p = cv2.norm(proj_pts[i], p2_p, cv2.NORM_L2)/len(p2_p)
            
            # เก็บข้อมูลค่า Error ของกล้องและโปรเจกเตอร์ลงตามคิวชื่อท่าทาง
            errors[p] = (err_c, err_p)
        return errors

    def calibrate_final(self, base_dir, selected_poses, output_file):
        # ฟังก์ชันทำ Calibration กล้องคู่ (Stereo Calib) เตาเต็มแมกซ์ แล้วกลั่นเอาโมเดลพารามิเตอร์สุดยอดออกมา (Final step)
        obj_pts, cam_pts, proj_pts, shape, _ = self.load_calib_data(base_dir, selected_poses)
        
        # เริ่ม Calibrate แต่ละเดี่ยวก่อน
        print("Calibrating Camera...")
        rc, mc, dc, _, _ = cv2.calibrateCamera(obj_pts, cam_pts, shape, None, None) 
        print("Calibrating Projector...")
        rp, mp, dp, _, _ = cv2.calibrateCamera(obj_pts, proj_pts, (SCREEN_WIDTH, SCREEN_HEIGHT), None, None)
        
        # Stereo Calibrate ขั้นตอนที่หนักที่สุด เพื่อผูกความสัมพันธ์ (E, F) ระหว่างกล้องหลัก และเลนส์ฉาย ให้เป็นหนึ่งเดียวกัน 
        print("Stereo Calibration...")
        ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            obj_pts, cam_pts, proj_pts, mc, dc, mp, dp, shape, flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        # ประมวลผลสร้างเรขาคณิตอิงแกนกล้องหลักเป็นที่ตั้ง (สมมติให้กล้องหลักเป็นศูนย์กลางของจักวาล 0,0,0)
        
        # 1. Camera Center (ประตูกล้อง คือใจกลางต้นกำหนด (Oc) เลยตั้งเป็น 0,0,0 )
        Oc = np.zeros((3, 1))
        
        # 2. Camera Rays (เส้นรังสี Nc ที่พาดผ่านจากทะลุเลนส์กล้อง ไปตามแต่พิสัยตกกระทบของแต่ละพิกเซลภาพ x,y)
        w, h = shape  # แก้บักความกว้างกระจกรองรับของดั้งเดิม ที่บางครั้งหน้าจอตั้งกับนอนสลับมา (w, h ไม่ใช่ h, w) 
        
        u, v = np.meshgrid(np.arange(w), np.arange(h)) # กริดเรขาคณิตจอทุกรูขุมขน
        fx, fy, cx, cy = K1[0,0], K1[1,1], K1[0,2], K1[1,2] # ดึงพารามิเตอร์มุมมองเลนส์ใน (Intrinsic)
        
        # แปลงแกนภาพ 2D ออกจากกันให้เป็นแกนกลางหักล้างรอยโค้งเลนส์ 
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy
        z_norm = np.ones_like(x_norm) # แกนดิ่ง Z พุ่งตรงยืดปริมณฑลดลยาว 1 ตลอดเวลา
        
        # ควบเป็น Ray เวกเตอร์สามทาง (Unit Vector)
        rays = np.stack((x_norm, y_norm, z_norm), axis=2)
        norms = np.linalg.norm(rays, axis=2, keepdims=True)
        rays /= norms # รีดให้เป็นบรรฐานขนาด 1 ป้องกันตัวแปรล้น 
        Nc = rays.reshape(-1, 3).T
        
        # 3. Projector Planes (สร้างสมการ 4 พิกัดของกำแพงแสงจากเครื่องโปรเจกเตอร์: แกนแนวตั้ง wPlaneCol และ แกนนอน wPlaneRow )
        wPlaneCol = np.zeros((SCREEN_WIDTH, 4))
        wPlaneRow = np.zeros((SCREEN_HEIGHT, 4))
        
        # พารามิเตอร์ในของโปรเจกเตอร์
        fx_p, fy_p = K2[0,0], K2[1,1]
        cx_p, cy_p = K2[0,2], K2[1,2]
        
        R_inv = R.T # เครื่องหมายทรานสโฟสหมุนกระจกเลนส์เทียบมุมสะท้อน
        C_p_cam = -R_inv @ T # จุดตั้งมั่นของเครื่องจานแสงบนแกนกล้อง
        
        # ฟังก์ชันรองเพื่อปั้นพิกัดเพลน (กำแพงแสง) ให้กลายเป็น 4 มุมสมบูรณ์อิงมุมฉากกับจุด Oc 
        def get_plane_from_proj_line(u_p, v_p_start, v_p_end, is_col=True):
            if is_col: # ถ้าเป็นเส้นแสงแนวตั้งฉากลงดิน
                p1_n = np.array([(u_p - cx_p)/fx_p, (v_p_start - cy_p)/fy_p, 1]).reshape(3,1)
                p2_n = np.array([(u_p - cx_p)/fx_p, (v_p_end - cy_p)/fy_p, 1]).reshape(3,1)
            else: # ถ้าเป็นเส้นเส้นระดับน้ำ
                p1_n = np.array([(v_p_start - cx_p)/fx_p, (u_p - cy_p)/fy_p, 1]).reshape(3,1)
                p2_n = np.array([(v_p_end - cx_p)/fx_p, (u_p - cy_p)/fy_p, 1]).reshape(3,1)
                
            r1 = R_inv @ p1_n
            r2 = R_inv @ p2_n
            
            # Cross Vector เพื่อหาทิศตั้งตรงตามการเบี่ยงทิศ 
            normal = np.cross(r1.flatten(), r2.flatten())
            normal /= np.linalg.norm(normal)
            d = -np.dot(normal, C_p_cam.flatten())
            
            return np.array([normal[0], normal[1], normal[2], d])

        # สกัดหน้าจอแนวนอนมาสร้างทีละแนวกำแพง
        for c in range(SCREEN_WIDTH):
            wPlaneCol[c, :] = get_plane_from_proj_line(c, 0, SCREEN_HEIGHT, is_col=True)
            
        # สกัดแนวตั้งแผ่นแสง
        for r in range(SCREEN_HEIGHT):
            wPlaneRow[r, :] = get_plane_from_proj_line(r, 0, SCREEN_WIDTH, is_col=False)
            
        # บันทึกความทุลักทุเลและสมการโลกคู่มลรัฐที่คำนวนมาทั้งมวล ลงไฟล์ข้อมูลเมตริกซ์ .mat ของ Scipy
        scipy.io.savemat(output_file, {
            "Nc": Nc,             # รังสีรับภาพของเซ็นเซอร์กล้องมือถือ
            "Oc": Oc,             # พิกัดกล้อง
            "wPlaneCol": wPlaneCol.T,  # กะทะทาบเส้นหน้าจอแนวดิ่ง
            "wPlaneRow": wPlaneRow.T,  # กะทะทาบแนวนอน
            "cam_K": K1,          # พารามิเตอร์เลนส์กล้องถ่ายรูป 
            "proj_K": K2,         # พารามิเตอร์เลนส์ไฟฉาย 
            "R": R,               # ค่าสัมประสิทธิ์หมุนโลก 
            "T": T                # ค่าเคลื่อนกล้อง
        })
        # สรุปงานพร้อมแจ้งความเพี้ยน (Error rate ที่ประเมิณ) ให้ดูต่างหน้า 
        messagebox.showinfo("Success", f"Calibration Saved to:\n{output_file}\nError: {ret:.4f}")

    # ---------------------------------------------------------
    # 3. CAPTURE SCAN (ขั้นตอนที่ 3: สแกนกวาดภาพจริงลงพื้นที่)
    # ---------------------------------------------------------
    def capture_scan(self, save_dir, silent=False):
        # เริ่มต้นกระบวนการสแกนแบบอัตโนมัติ โดยฉายนิรมิตภาพแสงบนพื้นผิวมหรสพ (วัตถุเป้าหมาย)
        
        # เริ่มต้นตั้งค่าโปรเจกเตอร์ (เช่น เปิดหน้าต่าง Fullscreen ไปยังขอรับรอง)
        self.init_projector()
        
        # สร้างชุดลวดลาย (Gray Code Patterns) ทั้งแนวตั้งและแนวนอนเก็บไว้ในตัวแปร P
        P = self.generate_patterns()
        
        # เตรียมลิสต์สำหรับเก็บภาพที่จะฉาย และภาพพื้นฐาน (ขาว/ดำ) ให้เป็นมาตรฐาน
        patterns = []
        white = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8) * PROJ_VALUE # ภาพ "ขาวล้วน" ใช้เปิดดูชิ้นงาน และฉายลอกลาย texture รูป 
        black = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8) # ทาหน้า "ดำล้วน" (เพื่อใช้ดูระดับ Noise หรือ Ambient light ถ่ายรบกวนในมืด)
        
        patterns.append(("01.bmp", white))
        patterns.append(("02.bmp", black))
        
        idx = 3 # ภาพสแกนตารางแทรกบิตเริ่มต้นรูปที่สาม
        # ขยำแอดภาพลายทาง (และด้านกลับ Inverse) ลงชามยักษ์ patterns สำหรับเตรียมการฉายต่อเนื่อง
        for j in range(2): 
            for pat in P[j]:
                pat_img = (pat * PROJ_VALUE).astype(np.uint8)
                inv_img = ((1-pat) * PROJ_VALUE).astype(np.uint8)
                
                if D_SAMPLE_PROJ > 1:
                    pat_img = cv2.resize(pat_img, (SCREEN_WIDTH, SCREEN_HEIGHT), interpolation=cv2.INTER_NEAREST)
                    inv_img = cv2.resize(inv_img, (SCREEN_WIDTH, SCREEN_HEIGHT), interpolation=cv2.INTER_NEAREST)
                    
                patterns.append((f"{idx:02d}.bmp", pat_img)); idx+=1
                patterns.append((f"{idx:02d}.bmp", inv_img)); idx+=1
        
        # สร้างโฟลเดอร์ดักรอเก็บรูปแบบ .bmp ไปพรางๆ (ใช้ .bmp จะซัพพอร์ตการบีบอัดไร้รอยต่อกว่า JPG)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imshow(self.window_name, white) # แกล้งเปิดจอสว่างให้ดูตระเวนวัตถุชั่วครู่
        cv2.waitKey(100)
        
        # ถ้าไม่ได้ตั้งค่าปิดแบบเงียบ (silent Mode สำหรับเครื่องกลอัตโนมัติ Turntable) ให้มีหน้าต่างแจ้งเตือนว่า "พร้อมทำงานแล้วนะพี่!"
        if not silent: 
            messagebox.showinfo("Step 3", f"Ready to scan.\nImages saved to: {save_dir}")
            
        # เปิดโรงฉาย ยิงปืนทีละช็อต
        for fname, img in patterns:
            cv2.imshow(self.window_name, img) # แถบโปรเจกเตอร์สลับ
            cv2.waitKey(200) # ให้เวลาโทรศัพท์ตั้งหน้าตั้งตารับแสงหน่อย และไม่วอนให้ภาพเบลอ 
            
            # บังคับโทรศัพท์กดแชะภาพ (หากล่มแจ้ง Timeout)
            if not self.trigger_capture(os.path.join(save_dir, fname)):
                if not silent:
                    messagebox.showerror("Error", "Timeout"); 
                self.close_projector(); return
        
        # เก็บของ จบงานฉายแสง ปิดหน้าจอผีหลอกตา
        self.close_projector()
        
        # รายงานสรุปส่งผลลัพธ์ว่าเซ็ตนี้ครบสมบูรณ์แบบ
        if not silent:
            messagebox.showinfo("Step 3 Done", "Scan Capture Complete.")

    # ---------------------------------------------------------
    # 4. GENERATE CLOUD (ขั้นตอนที่ 4: เสกจุดพอยต์ให้กลายเป็นกลุ่มเมฆ 3 มิติ)
    # ---------------------------------------------------------
    def generate_cloud(self, scan_dir, calib_file):
        # ฟังก์ชันคำนวณถอดรหัสจุด 3 มิติตามแกนหลักแสง และกระดานตาข่ายสวาท Calibrate อันเก่าที่เคสจัดทำไว้
        
        # ตรวจสอบว่ามีไฟล์ Calibration อยู่จริงไหม ถ้าไม่มีก็พับเสื่อกลับบ้านไปเลย 
        if not os.path.exists(calib_file):
            raise FileNotFoundError(f"Calibration file not found at {calib_file}")

        print(f"[Process] Processing {scan_dir} using {calib_file}...")
        
        # โหลดไฟล์ .mat (ที่ได้จากขั้นตอน Calibrate มาอย่างยากลำบาก) เข้ามาในโปรแกรม
        data = scipy.io.loadmat(calib_file)
        if 'Oc' not in data: # เช็คว่าไฟล์สมบูรณ์ไหม (ต้องมีค่า Oc หรือจุดศูนย์กลางกล้อง เป็นสัญลักษณ์ยืนยันชีพจรสมการ)
            raise ValueError("Calibration file missing 'Oc'.")
            
        # จัดแจงและสูบเสบียงตัวแปรต่างๆ ลงใน Dictionary เพื่อให้เรียกใช้ง่ายๆ เป็นหมวดหมู่
        calib_data = {
            "Nc": data["Nc"],               # รังสีของแสงจากกล้อง
            "Oc": data["Oc"],               # จุดศูนย์กลางกล้อง (ทวีป 0,0,0)
            "wPlaneCol": data["wPlaneCol"], # ระนาบแพงแสงของโปรเจกเตอร์แนวตั้ง
            "wPlaneRow": data["wPlaneRow"], # ระนาบแพงแสงของโปรเจกเตอร์แนวนอน
            "cam_K": data["cam_K"]          # ค่าคงที่เลนส์กล้อง
        }

        # Embed gray_decode exactly as it is in standalone
        # ฟังก์ชันรองที่ใช้เจาะและคลายปมรหัสแสง (Gray Decode) ที่ฉายแปะทับลงบนวัตถุ ว่าบิตพิกเซลไหนตรงกับกี่เส้น
        def gray_decode(folder, n_cols=1920, n_rows=1080):
            # เสาะหาไฟล์ภาพ .bmp หรือถ้าแอบเป็น .png ก็ไม่ว่า คุ้ยทั้งหมดในโฟลเดอร์รอยสแกน
            files = sorted(glob.glob(os.path.join(folder, "*.bmp")))
            if not files:
                files = sorted(glob.glob(os.path.join(folder, "*.png")))
                
            # ต้องมีอย่างมากมหาศาล ถ้ามีแค่ก้อนขยะไม่ถึง 4 ใบให้ตีกลับเป็น Error 
            if len(files) < 4:
                raise ValueError("Not enough images in folder to decode.")
                
            # ดึงภาพหน้าขาวอัด (01) และหลอกหน้าดำ (02) มาเป็นตัวเปิดทางเทียบสีวัดความชัดลึก (Contrast)
            img_white = cv2.imread(files[0], 0).astype(np.float32)
            img_black = cv2.imread(files[1], 0).astype(np.float32)
            
            height, width = img_white.shape
            
            # คำนวณขอบเขตและสร้างหน้ากาก Mask เพื่อคัดจุดที่ไม่ชัดเจนออกทิ้งขยะไป ไม่เอามาคิดแสง (พวกรอยหยัก เงา มืดเกินกว่ากล้องจะรับไหว)
            # Calculate Contrast and Noise Floor (สูตรลบขยะมืดตึบ)
            contrast = img_white - img_black # ระยะทิ้งตัวเงาสี 
            noise_floor = np.percentile(img_black, 95) # หาระดับลากจูงขยะที่ 95% ของกำแพงพิกเซลเม็ดดำ
            dynamic_range = np.max(contrast) # จุดที่แสงผ่องตัดกับดำปี๋ได้สูงที่สุด (คล้ายความชัดสุด)

            # ประยุกต์เกณฑ์ (Thresholding) ที่ปรับแปรผันตามรูปนั้นๆ ได้
            mask_shadow = img_white > (noise_floor * 1.5) # แสงสว่างจ้าต้องหนีกว่าชั้นเงาพิลึกอย่างน้อย 1.5 เท่า
            mask_contrast = contrast > (dynamic_range * 0.05) # มีอัตราคอนทราสหลุดเป้าแหว่งไปนิด 5% 

            # รวมทั้ง 2 สิทธิ์เป็นผู้รักษาการตีกรอบพื้นที่ใช้การได้ (ถัดจากนี้พื้นที่มืดหรืออับแสงจะไม่มีวันถูกประมวล 3 มิติ ได้ออกมาเริ่ดแน่นอน รูอาจจะกลวงๆ)
            valid_mask = mask_shadow & mask_contrast

            # จำนวนขั้นบันไดของแถบบิต ตามความกะทัดรัดของจอโปรเต็กเตอร์ 
            n_col_bits = int(np.ceil(np.log2(n_cols)))
            n_row_bits = int(np.ceil(np.log2(n_rows)))
            
            current_idx = 2 # สลับแฟ้มเริ่มถ่างข้อมูลรูปใบที่ 3 ขึ้นหน้าตัก
            
            # ฟังก์ชันตัวลูก ในการแหกบิตไขแสงเงาทะลวงไส้ (เหมือนตอน Load Calibrate) แบบถอดภาพ
            def decode_sequence(n_bits):
                nonlocal current_idx # ก้าวล้ำใช้ค่าหน้าตักเก่าวนๆไปมาได้
                gray_val = np.zeros((height, width), dtype=np.int32) # จอ 0 สมบูรณ์
                
                # ฟาดการถอดหรัสทีละเส้นๆ จนครบจำนวนบิตที่ระบุ (ทะลวงหน้าทะลุหลัง)
                for b in range(n_bits):
                    if current_idx >= len(files): break
                    
                    # ปล่อยคู่ภาพปกติ (p) กับผกผัน Inverse (i) มาซัดกันสว่าง
                    p_path = files[current_idx]; current_idx += 1
                    i_path = files[current_idx]; current_idx += 1
                    
                    img_p = cv2.imread(p_path, 0).astype(np.float32)
                    img_i = cv2.imread(i_path, 0).astype(np.float32)

                    # สแกนทับ ถ้าหน้าสว่างกว่าหลัง ก็มอบตราตั้งให้บิตตานั้นๆ เป็น '1' 
                    bit = np.zeros((height, width), dtype=np.int32)
                    bit[img_p > img_i] = 1

                    # ดันบิตสวมเกราะถักทอ (Bit Shifting) ลงตะกร้าตานึงเพื่อต่อยอดเป็นตัวเลข ฐานรหัส Gray
                    gray_val = np.bitwise_or(gray_val, np.left_shift(bit, (n_bits - 1 - b)))
                    
                # เมื่อซดน้ำบิตหมดจอ ก็ต้องแปลงร่างรหัส Gray Code เหล่านั้น คืนสภาพกลายเป็นเลขหน้ากระดาน Binary ธรรมดา (เหมือนพิกัดพิกเซลนั่นแล)
                mask = np.right_shift(gray_val, 1)
                while np.any(mask > 0): # คลายมนต์ด้วย XOR ให้ถึพจนหมดจรด 
                    gray_val = np.bitwise_xor(gray_val, mask)
                    mask = np.right_shift(mask, 1)
                    
                return gray_val # แผ่กระจุยเลขพิกเซลจริง ส่งคืน

            print("Decoding Columns...")
            col_map = decode_sequence(n_col_bits)  # ถอดรหัสตำแหน่งแถวตั้งบนลูกกระเดือกวัตถุ
            print("Decoding Rows...")
            row_map = decode_sequence(n_row_bits)  # ถอดแนวนอนทาลา
            
            # คืนค่าทั้งหมดพ่วงด้วยแถมให้คือ ผิวดิบหน้ากระดานขาว Texture ไว้ไปประกอบสีโมเดล 3D
            return col_map, row_map, valid_mask, cv2.imread(files[0]) 
            
        # Embed reconstruct_point_cloud exactly as it is in standalone
        # ฟังก์ชันรองถัดไป ทำความเข้าใจและฟัน 3D (3D Triangulation Ray-Intersection) ตกตอดจากข้อมูล 2D ตะกี้ 
        def reconstruct_point_cloud(col_map, row_map, mask, texture, calib): 
            print("Reconstructing 3D points...")
            
            Nc = calib["Nc"] # รังสีแสงที่กล้องเล็งไว้เป๊ะๆ
            Oc = calib["Oc"] # โหมลงเซ็นเซอร์กลางเป้า
            wPlaneCol = calib["wPlaneCol"] # กำแพงโปรเจกเตอร์แนวตั้งที่สาดทับไป
            
            if wPlaneCol.shape[0] == 4: wPlaneCol = wPlaneCol.T # กลับข้างให้เลขตังตรงเป๊ะ
            
            h, w = col_map.shape 
            
            # กดทุกอย่างให้แบลงๆ (Flatten) จะคำนวณไวกว่าซ้อนชั้นอาร์เรย์เป็นลิสต์มิติมหาชัย
            col_flat = col_map.flatten()
            mask_flat = mask.flatten()
            tex_flat = texture.reshape(-1, 3) # เตรียมสี (B G R ลำดับ OpenCV ปกติ) ของพิกเซลนั้นๆ
            
            # เลือกใช้แค่จุดที่ทะลุเกราะ Mask มาในรอบคัดเลือกขยะทิ้งนั่นแหละ เอามาปั่น
            valid_indices = np.where(mask_flat)[0]
            print(f"Processing {len(valid_indices)} valid pixels...")
            
            # คำนวณหาทิศทางของแสง (Rays) สำหรับลากรังสีเชื่อมกล้องเข้าไปหาทุกพิกเซล 
            if Nc.shape[1] == h * w:
                rays = Nc[:, valid_indices] # ดึงสายลากยาวได้ทันที
            else:
                # ถ้าเซ็ตเก่าไม่มี Nc สำเร็จรูปมา ก็ปลุกเสกเส้นสร้างมันขึ้นใหม่จาก Matrix วงในแกน K เองหน้างาน
                K = calib["cam_K"]
                fx, fy = K[0,0], K[1,1]
                cx, cy = K[0,2], K[1,2]
                
                # แปลงร่างลำดับแบนๆให้คืนทรง (h,w) หาแถวช่องง่ายขี้น
                y_v, x_v = np.unravel_index(valid_indices, (h, w))
                x_n = (x_v - cx) / fx
                y_n = (y_v - cy) / fy
                z_n = np.ones_like(x_n)
                
                rays = np.stack((x_n, y_n, z_n)) # เรียงพิกัด 3 ด้านควบรวม
                norms = np.linalg.norm(rays, axis=0) # ลดให้เส้นแสงสั้น 1 หน่วย Unit
                rays /= norms
                
            # ดึงสมการระนาบเพลน (Plane) ของโปรเจกเตอร์ที่หน้าชนพอดีตรงเป๊ะกับพิกัดเส้นตารางที่เราถอดรหัสได้ตะกี้ 
            proj_cols = col_flat[valid_indices]
            # ห้ามล้นหน้าจอสุดๆนะ เผื่อมีบัก Clip มันไว้ทึ่ขอมสระ 1920-1
            proj_cols = np.clip(proj_cols, 0, wPlaneCol.shape[0] - 1) 
            
            # ล้วงจับสมการระแวดระวัง
            planes = wPlaneCol[proj_cols, :]
            
            # N(x,y,z) และ d สำหรับฟอร์ม Plane equation : Ax+By+Cz+D = 0
            N = planes[:, 0:3].T    # เวกเตอร์แกนเฉียงแนวตั้งฉากของหน้าเพลน (Normal Vector of Plane)
            d = planes[:, 3]        # ระยะห่างพื้นฐานของระนาบจากจุดกำเนิด (Distance to origin)
            
            # คำนวณหาจุดตัด (Intersection Point) ระหว่าง "เส้นแสง Ray ของกล้อง" กับ "กำแพงแสง Plane ของโปรดเจ็กเตอร์"
            # ใช้สมการบรรจบหาตั้มเวกเตอร์ $t$:
            # $t = -(N^T \cdot O_c + d) / (N^T \cdot \text{ray})$ 
            denom = np.sum(N * rays, axis=0) # ระยะบรรจบทับกัน ตัวส่วน 
            numer = np.dot(N.T, Oc).flatten() + d # ตัวคำนวณด้านบน 
            
            # สกีนป้องกันความล่มจมของการหารด้วยศูนย์ หรือเส้นรังสีขนาดขนานกับระนาบจนไม่สามารถจุดตัด 
            valid_intersect = np.abs(denom) > 1e-6 
            t = -numer[valid_intersect] / denom[valid_intersect] # จัดสรรผลลัพธ์หาตัว t ซะ
            
            # ถลุงรังสีส่วนที่ดีงามแล้วจับโยนระยะตั้ม t พุ่งออกไปหาพิกัดโลกเลย! 
            rays_valid = rays[:, valid_intersect]
            # พิกัด 3D ที่แท้ทรูบนแผ่นโลก: $P = O_c + t \cdot \text{ray}$
            P = Oc + rays_valid * t 
            
            # แถม! กระตุกสีพื้นผิว (C) (Color) นำขึ้นเรือมาตกแต่งจุดแต่ละจุด ให้ตรงจุดบรรจบ
            C = tex_flat[valid_indices[valid_intersect]]
            
            return P.T, C # ส่งพิกัดและพู่กันสีกลับสู้โลก
            
        # -----------------------------------------------------------------------------------
        # กลับเข้าคิวหลัก 
        # ลำดับที่ 1 ถอดรหัส 2D มาเลยลูกพี่ 
        c_map, r_map, mask_out, texture_out = gray_decode(scan_dir)
        
        # ลำดับที่ 2 ลามปามไป 3D พิกเซลล่องขยะ
        points, colors = reconstruct_point_cloud(c_map, r_map, mask_out, texture_out, calib_data)
        
        # ลำดับที่ 3 สร้างและบันทึกไฟล์ .ply (รูปแบบมาตรฐานสากลของไฟล์ประเภทเสี้ยน Point Cloud มีสี)
        # ตีตารับรองนามสกุล
        ply_name = os.path.basename(scan_dir) + ".ply"
        out_path = os.path.join(scan_dir, ply_name)
        
        print(f"Saving {len(points)} points to {out_path}...")
        
        # ดำนาเขียนเปิดไฟล์แบบเก๋าๆ 
        with open(out_path, 'w') as f:
            # เขียนหน้าปกบอกรุ่น หัวข้อ (Header) ของไฟล์ PLY รูปแบบ ASCII สามารถเปิดอ่านได้ด้วยตา
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
            
            # ทุบดินเขียนพิกัดลงทีละห้วงจุดพร้อมสีประกบคู่ (Loop วนไปทุกๆพอยนต์)
            for i in range(len(points)):
                p = points[i] # พิกัด 3 ตัว
                c = colors[i] # สี 3 กษัตริย์
                
                # มีทริคนิดนึง: กล้องถ่ายภาพ OpenCV ให้สีกลับหัวเป็น B-G-R แต่ PLY อยากได้ R-G-B
                # ดังนั้น เราต้องสลับหลักการเทสีทุ้งตอนปริ้นท์มัน (c[2] = R, c[1] = G, c[0] = B)
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[2]} {c[1]} {c[0]}\n")
                
        # ปรมมือจบ แจ้งผลสวยงาม
        print(f"[Success] Generated {out_path}")
