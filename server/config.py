import os
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================

# --- Folder Structure (โครงสร้างโฟลเดอร์) ---
# โฟลเดอร์หลักสำหรับจัดเก็บข้อมูลการสแกน โดยสร้างชื่อตามวันที่ปัจจุบัน (เช่น 25_02_2026_3Dscan)
DEFAULT_ROOT = os.path.join(os.getcwd(), f"{datetime.now().strftime('%d_%m_%Y')}_3Dscan")

# --- Hardware Settings (การตั้งค่าฮาร์ดแวร์) ---
# ตำแหน่งหน้าจอของ Projector ในแนวนอน (เริ่มต้นที่ 1920 พิกเซล หมายความว่าเป็นจอที่ 2)
SCREEN_OFFSET_X = 1920  
# ความกว้างของหน้าจอ Projector (1920 พิกเซล)
SCREEN_WIDTH = 1920     
# ความสูงของหน้าจอ Projector (1080 พิกเซล)
SCREEN_HEIGHT = 1080    
# ค่าความสว่างของแสงที่ฉายจาก Projector (0-255)
PROJ_VALUE = 200        
# อัตราส่วนการลดขนาดภาพ (Downsampling) เพื่อเพิ่มความเร็ว (1 คือไม่ลด, ค่ามากกว่า 1 คือลดขนาดภาพลง)
D_SAMPLE_PROJ = 1       

# --- Checkerboard Settings (การตั้งค่ากระดานหมากรุกสำหรับ Calibrate) ---
# จำนวนจุดตัดมุมในตารางหมากรุก แถว (Inner corners)
CHECKER_ROWS = 7        
# จำนวนจุดตัดมุมในตารางหมากรุก คอลัมน์ (Inner corners)
CHECKER_COLS = 7
# ขนาดของช่องสี่เหลี่ยมในกระดานหมากรุก (หน่วยเป็นมิลลิเมตร)
SQUARE_SIZE = 35.0      
