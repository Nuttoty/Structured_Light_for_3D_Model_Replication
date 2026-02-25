import os
import glob
import copy
import numpy as np
import open3d as o3d

# ==========================================
# PROCESSING LOGIC (Open3D)
# ==========================================
class ProcessingLogic:
    # คลาสสำหรับประมวลผลโมเดล 3 มิติ (Point Cloud และ Mesh) โดยใช้ไลบรารี Open3D
    @staticmethod
    def _load_pcd(input_data):
        # ฟังก์ชันภายในสำหรับตรวจสอบและโหลดไฟล์ Point Cloud
        if isinstance(input_data, str): # ถ้าข้อมูลที่รับมาเป็นข้อความ (พาธไฟล์)
            if not os.path.exists(input_data): # ถ้าหาไฟล์ตามพาธไม่เจอ
                raise FileNotFoundError(f"Input file not found: {input_data}") # แจ้งข้อผิดพลาดว่าไม่พบไฟล์
            # อ่านและคืนค่าข้อมูล Point Cloud จากไฟล์นั้นด้วย Open3D
            return o3d.io.read_point_cloud(input_data)
            
        # ถ้าไม่ได้เป็นข้อความ (สมมุติว่าเป็น Object Point Cloud อยู่แล้ว) จะคืนค่าเดิมกลับไป
        return input_data

    @staticmethod
    def remove_background(input_data, output_path=None, distance_threshold=50, ransac_n=3, num_iterations=1000, return_obj=False):
        # ฟังก์ชันสำหรับลบพื้นหลัง/กำแพงด้านหลัง (Background Remove) ออกจากโมเดล 3D
        print(f"[BG Remove] Processing...")
        
        # โหลดไฟล์ Point cloud เข้ามาประมวลผล
        pcd = ProcessingLogic._load_pcd(input_data)
        
        # ถ้ารูปทรง 3D ที่โหลดมาไม่มีจุดพิกัดใดๆเลย
        if not pcd.has_points():
            raise ValueError("Point cloud is empty.") # แจ้งข้อผิดพลาด

        # ใช้เทคนิค Segment Plane (หาระนาบที่ใหญ่ที่สุด) ถือว่าระนาบนใหญ่นั่นคือกำแพงพื้นหลัง
        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                 ransac_n=ransac_n,
                                                 num_iterations=num_iterations)
        
        # เลือกลบเฉพาะ inliers (จุดที่เป็นของระนาบ/กำแพง) เก็บส่วนที่เหลือ (Outliers) ซึ่งก็คือวัตถุหลัก (Object) ไว้ (invert=True)
        object_cloud = pcd.select_by_index(inliers, invert=True)
        
        # แสดงผลจำนวนพิกัดจุดก่อนลบ กับหลังลบ
        print(f"[BG Remove] Original: {len(pcd.points)}, Remaining: {len(object_cloud.points)} pts")
        
        # ถ้ามีการกำหนดตำแหน่งที่จะบันทึกไฟล์ (Output path) ไว้
        if output_path:
            o3d.io.write_point_cloud(output_path, object_cloud) # บันทึกเป็นไฟล์ใหม่
            print(f"[BG Remove] Saved to {output_path}")
            
        return object_cloud if return_obj else None # คืนค่าข้อมูลออบเจ็กต์กลับไป (เว้นแต่ไม่ต้องการให้คืนค่า None)

    @staticmethod
    def remove_outliers(input_data, output_path=None, nb_neighbors=20, std_ratio=2.0, return_obj=False):
        # ฟังก์ชันสำหรับลบจุดรบกวนระยะห่างหรือฝุ่นผง (Outlier Removal)
        print(f"[Outlier] Processing...")
        pcd = ProcessingLogic._load_pcd(input_data) # โหลดไฟล์
        
        if not pcd.has_points():
            raise ValueError("Point cloud is empty.")

        # ใช้คำสั่งลบจุดที่อยู่ห่างผิดปกติด้วยสถิติ โดยคัดแยกตามจำนวนเพื่อนบ้าน (Neighbors) และค่าเบี่ยงเบนมาตรฐาน
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        
        # คัดกรองเหลือแต่จุดที่ผ่านเกณฑ์เท่านั้นมาใช้งาน
        inlier_cloud = pcd.select_by_index(ind)
        
        print(f"[Outlier] Keeping: {len(inlier_cloud.points)} pts")
        
        # บันทึกเป็นไฟล์ 3D ลงคอมพิวเตอร์ถ้ามีพาธ
        if output_path:
            o3d.io.write_point_cloud(output_path, inlier_cloud)
            print(f"[Outlier] Saved to {output_path}")

        return inlier_cloud if return_obj else None

    @staticmethod
    def preprocess_point_cloud(pcd, voxel_size):
        # ฟังก์ชันจัดเตรียมข้อมูลโมเดล (Downsample + Normals + FPFH Features) ก่อนนำไปเชื่อม (Merge)
        
        # 1. ลดความละเอียดโมเดลลง (Downsample) ให้อยู่ในกรอบ Voxel เพื่อประหยัดเวลาคำนวณ
        pcd_down = pcd.voxel_down_sample(voxel_size)
        
        # 2. คำนวณหาทิศทางพื้นผิว (Normals) สำหรับช่วยในการจับคู่โมเดล
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            
        # 3. คำนวณคุณลักษณะเฉพาะพิกัด FPFH (Fast Point Feature Histograms) สำหรับการทำ RANSAC แบบสมบุกสมบัน
        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            
        return pcd_down, pcd_fpfh # คืนค่าโมเดลแบบหยาบ และโมเดลคุณสมบัติ

    @staticmethod
    def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
        # ฟังก์ชันทำ Global Registration (จัดตำแหน่งรวม 2 โมเดล ให้หันเข้าหากันคร่าวๆ แบบ RANSAC)
        distance_threshold = voxel_size * 1.5
        
        # ใช้ RANSAC ร่วมกับ FPFH เพื่อคาดเดาจุดที่เข้าคู่ (Matching) กันได้มากที่สุด
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
            
        return result

    @staticmethod
    def merge_pro_360(input_folder, output_path, voxel_size=0.02):
        # ฟังก์ชันหลักสำหรับรวมกลุ่มโมเดล 3D ที่ได้จากการสแกน 360 องศา (หลายมุม) เข้าด้วยกันตามลำดับ
        print(f"[Merge 360] Loading clouds from {input_folder}...")
        
        # หาไฟล์ .ply ทั้งหมดในโฟลเดอร์ และเรียงตามชื่อ
        ply_files = sorted(glob.glob(os.path.join(input_folder, "*.ply")))
        if len(ply_files) < 2:
            raise ValueError("Need at least 2 .ply files to merge.") # ต้องมีอย่างน้อย 2 โมเดลถึงจะเชื่อมกันได้
            
        pcds = []
        for path in ply_files:
            # โหลดไฟล์แต่ละโมเดลเข้ามาในลูปเพื่อจัดเก็บเรียงเป็น List (pcds)
            pcd = o3d.io.read_point_cloud(path)
            pcds.append(pcd)
            
        print(f"[Merge 360] Loaded {len(pcds)} clouds. Running Sequential Registration (New360 Logic)...")
        
        # ให้โมเดลตั้งต้นคือ โมเดลแรก (Frame 0) เป็นตัวฐาน (Accumulator)
        merged_cloud = copy.deepcopy(pcds[0])
        
        # เก็บประวัติการประมวลผลเมทริกซ์การแปลงตำแหน่งสะสมของทุกกรอบ (Current Global Transform)
        max_accum_T = np.identity(4) 
        
        # วนรอบเทียบและต่อโมเดลทีละคู่ (โมเดลที่ 1 เทียบ 0, โมเดลที่ 2 เทียบ 1,...) เรื่อยมา 
        for i in range(1, len(pcds)):
            print(f"[Merge 360] Aligning Scan {i} -> Scan {i-1}...")
            source = pcds[i]      # โมเดลล่าสุด (ขยับเข้าหา)
            target = pcds[i-1]    # โมเดลก่อนหน้า (ตั้งนิ่งๆ)
            
            # 1. Preprocess เตรียมข้อมูลทั้งคู่ (ลดขนาดภาพ + หา Normals)
            source_down, source_fpfh = ProcessingLogic.preprocess_point_cloud(source, voxel_size)
            target_down, target_fpfh = ProcessingLogic.preprocess_point_cloud(target, voxel_size)
            
            # 2. ให้ Open3D พยายามเดาตำแหน่งซ้อนทับภาพกว้างๆ ก่อน (Global RANSAC)
            ransac_result = ProcessingLogic.execute_global_registration(
                source_down, target_down, source_fpfh, target_fpfh, voxel_size)
            
            # 3. ให้ Open3D ทำการขยับซ้อนทับแบบแม่นยำขึ้นจากระยะเดิม (Local ICP Refinement จุดต่อระนาบ Point-to-Plane)
            icp_result = o3d.pipelines.registration.registration_icp(
                source_down, target_down, voxel_size, ransac_result.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            
            # ดึงสมการ (Matrix) ขยับตำแหน่งความสัมพัทธ์ระหว่าง i กับ i-1 มาเก็บ
            T_local = icp_result.transformation 
            
            # 4. แปลงให้เป็นความสัมพันธ์จาก i ขยับลงมาเทียบกับโมเดลตั้งต้น 0 ที่สุด เพื่อให้ทุกชิ้นอยู่เวทีเดียวกัน
            max_accum_T = np.dot(max_accum_T, T_local)
            
            # 5. สั่งขยับและประกอบรวมกันเข้ากับเวทีตั้งต้น
            pcd_temp = copy.deepcopy(source) 
            pcd_temp.transform(max_accum_T) # เปลี่ยนตำแหน่งโมเดลล่าสุดแล้วขยับซ้อน
            merged_cloud += pcd_temp        # เทรวมกัน
            
        print("[Merge 360] Post-processing (Downsample + Outlier removal)...")
        # เอาโมเดลใหญ่ทั้งหมดที่เสร็จแล้ว มารดลดขนาดความละเอียดลงรอบสุดท้ายกันคอมกระตุก
        pcd_combined_down = merged_cloud.voxel_down_sample(voxel_size=voxel_size)
        
        # ปัดฝ้ายกรองจุดเสียรอบสุดท้ายของการรวมภาพ (Outlier Removal)
        cl, ind = pcd_combined_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd_final = pcd_combined_down.select_by_index(ind)
        
        # คำนวณองศาผิว Normals ล่าสุดให้กับโมเดลใหญ่
        pcd_final.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        
        # บันทึกไฟล์ที่รวมภาพเรียบร้อยสมบูรณ์แล้วส่งออกเป็น PLY
        o3d.io.write_point_cloud(output_path, pcd_final)
        print(f"[Merge 360] Saved merged cloud to {output_path}")

    @staticmethod
    def reconstruct_stl(input_path, output_path, mode="watertight", params=None):
        # ฟังก์ชันใช้สร้างเป็นแผ่นตะแกรง 3D หรือทึบสมบูรณ์ (STL from Point Cloud) เหมาะแก่งานพิมพ์ 3 มิติ
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        print(f"[Recon] Loading {input_path}...")
        pcd = o3d.io.read_point_cloud(input_path) # อ่านไฟล๋พอยต์คราวด์มา
        
        if not pcd.has_points():
            raise ValueError("Point cloud is empty.")
            
        # สร้างเส้น Normals เบื้องต้น (ทิศทางผิวหน้า) ทันทีถ้าไฟล์ต้นทางไม่มีมาให้ ไม่งั้นฉาบแสงไม่ได้
        if not pcd.has_normals():
            print("[Recon] Estimating normals...")
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
            # เช็คและหมุนเส้น Normals ให้ทิศทางเดียวกันทั้งหมด
            pcd.orient_normals_consistent_tangent_plane(100)
            
        mesh = None
        if mode == "watertight":
            # สร้างการถักลวด 3 มิติแบบปิดรูรั่วและมิดชิด (Poisson Surface Reconstruction)
            depth = int(params.get("depth", 10)) # ดึงค่าความลึก/ความละเอียด 
            if depth > 16:
                raise ValueError(f"Depth {depth} is too high! Maximum recommended is 12-14. >16 will freeze your PC.")
            
            print(f"[Recon] Poisson Reconstruction (depth={depth})...")
            # สร้าง Mesh จาก Point โดยตรงด้วยสูตรสมการ Poisson
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, linear_fit=False)
            
            # ตัดเนื้อส่วนเกินหรือพิกัดลวงที่ Open3D พยายามยืดปิดรูหลอกๆ คืนออก
            densities = np.asarray(densities)
            mask = densities < np.quantile(densities, 0.02) # ตัดขอบที่มีความหนาแน่นต่ำทิ้ง
            mesh.remove_vertices_by_mask(mask)
            
        elif mode == "surface":
            # วิธีถมสร้าง 3D อีกแบบคือ Ball Pivoting (ปั้นลูกบอลกลิ้งเชื่อมจุด) ปิดรูไม่ได้ แต่เก็บรายละเอียดบนผิวดีกว่า
            radii_str = params.get("radii", "1,2,4")
            try:
                # คำนวณหาระยะความหนาแน่นเฉลี่ยระหว่างแต่ละจุดรอบๆก่อน ว่าจุดส่วนใหญ่ในงานนี้ใหญ่แค่ไหน
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                
                # นำขนาดที่ได้ไปคูณกับระดับค่าสัมประสิทธิ์ใน UI (เช่น 1, 2, 4) แปลงเป็นลิสต์ตัวคูณขนาดของลูกบอลที่จะรับมาเชื่อม
                multipliers = [float(x) for x in radii_str.split(',')]
                radii = [avg_dist * m for m in multipliers]
                print(f"[Recon] Ball Pivoting (radii={radii})...")
                
                # สร้าง Mesh ด้วยขนาดลูกบอลทบยอดหลายเบอร์
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii))
            except Exception as e:
                raise ValueError(f"Invalid radii parameters: {e}")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ถ้าถักสร้าง Mesh ล้มเหลวไม่มีโมเดลแสดง
        if len(mesh.vertices) == 0:
            raise ValueError("Generated mesh is empty.")

        # ประมวลและใส่พื้นผิวเสมือนก่อนเขียนลงไฟล์
        print("[Recon] Computing normals and saving...")
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(output_path, mesh) # เซพไฟล์ .stl (หรือนามสกุลอื่นที่ Open3D รองรับ)
        print(f"[Recon] Saved STL to {output_path}")

    @staticmethod
    def mesh_360(input_path, output_path, depth=10, density_trim=0.01, orientation_mode="tangent"):
        # ฟังก์ชันสร้างและปรับแต่ง Mesh เฉพาะสำหรับการประมวลผลโมเดลจากการสแกนรอบตัว (360 องศา)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        print(f"[360 Mesh] Loading {input_path}...")
        pcd = o3d.io.read_point_cloud(input_path) 
        
        if not pcd.has_points():
            raise ValueError("Point cloud is empty.")
            
        # 1. คำนวณทิศผิวหน้า Normal เริ่มต้นให้กับพอยต์คลาวด์
        print("[360 Mesh] Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # 2. ปรับการตั้งค่าเรียงตัวทิศทาง Normal ของผิวหน้าใหม่ ป้องกันอาการผิวปลิ้นในออกนอก
        print(f"[360 Mesh] Re-orienting normals (Mode: {orientation_mode})...")
        
        if orientation_mode == "radial":
            # กรณีแบบ Radial (มุมรัศมีดาว) จะพุ่งทิศทางชี้เข้าสู่ศูนย์กลางแกนตลอด เหมาะกับงานหมุน
            center = pcd.get_center() # หาจุดศูนย์กลางของโมเดล
            pcd.orient_normals_towards_camera_location(center) # บังคับปลาย Normal ทั้งหมดให้ชี้เข้าหากลาง
            
            # เมื่อทิศชี้นั้นคือ "ด้านใน" จึงไปสลับค่าคูณผลติดลบ เพื่อพลิกพื้นผิวทั้งหมดให้หันหน้าออกมา "ด้านนอก" แทนนั่นเอง
            pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals) * -1.0)
            print("[360 Mesh] Radial orientation applied (Outwards).")
            
        else: # "tangent" กรณีปกติ 
            try:
                # พยายามหันให้สอดคล้องต่อกันเอง (Graph-based Consistency)
                pcd.orient_normals_consistent_tangent_plane(100)
                print("[360 Mesh] Consistent tangent plane orientation applied.")
            except Exception as e:
                # ถ้าล้มเหลวก็ย้อนกลับไปทำท่า Radial ให้แทน
                print(f"[360 Mesh] Warning: Tangent plane failed ({e}). Fallback to radial.")
                center = pcd.get_center()
                pcd.orient_normals_towards_camera_location(center)
                pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals) * -1.0)

        # 3. ขึ้นรูปเนื้อ Mesh ให้เติมเต็มโมเดล
        print(f"[360 Mesh] Poisson Reconstruction (depth={depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, linear_fit=False)
            
        # 4. หั่นส่วนเกิน (Optional) ถ้าค่าติดกว่า 0 จะลบก้อนเนื้อปูดที่ระบบสร้างขึ้นมั่วๆบริเวณกลวงทิ้งได้ระดับหนึ่ง
        if density_trim > 0.0:
            print(f"[360 Mesh] Trimming low density vertices (threshold={density_trim})...")
            densities = np.asarray(densities)
            threshold = np.quantile(densities, density_trim) 
            mask = densities < threshold
            mesh.remove_vertices_by_mask(mask) # ล้างยอดพิกัดที่ไม่น่าจำเป็น
        else:
            print("[360 Mesh] Density trim is 0.0 -> Keeping watertight result.")
        
        # 5. ล้างการประมวลผลผิวสุดท้ายก่อนส่งออก
        mesh.compute_vertex_normals()
        
        # 6. บันทึกพิมพ์ไฟล์โมเดลเสร็จสมบูรณ์ออกเป็น 3D โมเดล (อาทิ .stl) โชว์ที่เครื่องคอมพิวเตอร์และลบการคำนวณทิ้งไว้
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"[360 Mesh] Saved to {output_path}")

