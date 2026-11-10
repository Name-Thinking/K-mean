"""
=============================================================================
โครงงาน: ระบบลงทะเบียนใบหน้าลูกค้าอัตโนมัติ (Easy Customer Enrollment Studio)
ไฟล์: FaceTrain.py
คำอธิบาย:
    โปรแกรมบันทึกและลงทะเบียนใบหน้าลูกค้า/สมาชิกใหม่เข้าสู่ระบบอย่างง่ายดาย:
    - ตรวจจับใบหน้าอัตโนมัติด้วย YuNet (ไม่ต้องเล็งหน้าในกรอบแดงตายตัว)
    - ตรวจสอบคุณภาพความคมชัด แสงสว่าง และมุมมองหน้าตรงก่อนบันทึกอัตโนมัติ
    - เก็บภาพพอร์ตเทรตสมบูรณ์ขนาด 180x240 (เต็มศีรษะ ผม หน้าผาก คาง) และ 128-D Embedding 5 มุมภายใน 2 วินาที
    - รองรับชื่อภาษาไทย/อังกฤษ และบันทึกโปรไฟล์ลง customers/ พร้อมใช้งานทันที
=============================================================================
"""

import os
import sys
import time
import argparse
import tkinter as tk
from tkinter import simpledialog
from typing import Optional, List, Tuple

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from face_engine import (
    FaceModelManager,
    CustomerDatabase,
    ThaiHudRenderer,
    assess_face_quality,
    extract_clean_portrait,
    open_webcam,
)


def prompt_customer_info_gui(default_name: str = '') -> Tuple[Optional[str], Optional[int]]:
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        name = simpledialog.askstring(
            "ลงทะเบียนใบหน้าลูกค้าใหม่",
            "กรุณากรอกชื่อลูกค้า / ชื่อพนักงาน (ภาษาไทยหรืออังกฤษ):",
            initialvalue=default_name,
            parent=root
        )
        root.destroy()
        if not name or not name.strip():
            return None, None
        return name.strip(), None
    except Exception:
        print("\n" + "=" * 55)
        print("  ลงทะเบียนใบหน้าลูกค้าใหม่ (Customer Enrollment)")
        print("=" * 55)
        name = input("กรุณากรอกชื่อลูกค้า (เว้นว่างเพื่อยกเลิก): ").strip()
        if not name:
            return None, None
        return name, None


def run_enrollment_on_camera(
    cap: cv2.VideoCapture,
    customer_name: str,
    target_samples: int = 5,
    auto_capture: bool = True,
    group_id: Optional[int] = None,
    db: Optional[CustomerDatabase] = None,
    window_name: str = "Customer Enrollment  (SPACE=Manual | Q=Cancel)"
) -> bool:
    models = FaceModelManager.get_instance()
    hud = ThaiHudRenderer()
    if db is None:
        db = CustomerDatabase()

    captured_portraits: List[np.ndarray] = []
    captured_aligned: List[np.ndarray] = []
    last_capture_time = 0.0
    capture_interval = 0.45
    flash_frames = 0

    print(f"[Enroll] กำลังเริ่มบันทึกใบหน้าของ '{customer_name}' (ต้องการ {target_samples} ภาพ)...")

    while len(captured_portraits) < target_samples:
        ret, frame = cap.read()
        if not ret:
            print("[Error] ไม่สามารถอ่านภาพจากกล้องได้")
            break

        frame_disp = cv2.flip(frame, 1)
        h, w = frame_disp.shape[:2]

        faces = models.detect_faces(frame_disp, score_threshold=0.65)
        text_items = []

        overlay = frame_disp.copy()
        cv2.rectangle(overlay, (0, 0), (w, 85), (25, 30, 38), -1)
        cv2.addWeighted(overlay, 0.80, frame_disp, 0.20, 0, frame_disp)

        text_items.append({
            'text': f"ลงทะเบียนลูกค้า: {customer_name}   ({len(captured_portraits)}/{target_samples} ภาพ)",
            'pos': (20, 12), 'size': 24, 'color': (255, 255, 255), 'bold': True
        })

        status_msg = "กรุณามองกล้อง — ระบบกำลังตรวจจับใบหน้าอัตโนมัติ..."
        status_color = (0, 200, 255)
        best_face = None

        if faces is not None and len(faces) > 0:
            areas = faces[:, 2] * faces[:, 3]
            best_face = faces[np.argmax(areas)]
            fx, fy, fw, fh = map(int, best_face[:4])
            quality = assess_face_quality(frame_disp, best_face)

            if quality.is_valid:
                box_color = (80, 220, 100)
                status_msg = f"✓ {quality.reason_th} — กำลังบันทึกอัตโนมัติ..."
                status_color = (80, 255, 120)

                for lm_idx in range(5):
                    lx = int(best_face[4 + lm_idx * 2])
                    ly = int(best_face[5 + lm_idx * 2])
                    cv2.circle(frame_disp, (lx, ly), 3, (0, 255, 255), -1)

                now = time.time()
                if auto_capture and (now - last_capture_time >= capture_interval):
                    portrait = extract_clean_portrait(frame_disp, best_face)
                    aligned = models.align_and_crop(frame_disp, best_face)
                    captured_portraits.append(portrait)
                    captured_aligned.append(aligned)
                    last_capture_time = now
                    flash_frames = 3
                    print(f"  -> เก็บภาพตัวอย่างที่ {len(captured_portraits)}/{target_samples} (Quality={quality.score:.2f})")
            else:
                box_color = (0, 165, 255)
                status_msg = f"! {quality.reason_th}"
                status_color = (0, 200, 255)

            hud.draw_rounded_rect(frame_disp, (fx, fy), (fx + fw, fy + fh), box_color, thickness=3, radius=18)

        text_items.append({
            'text': status_msg,
            'pos': (20, 48), 'size': 19, 'color': status_color, 'bold': False
        })

        bar_w = int(w * 0.6)
        bar_x = (w - bar_w) // 2
        bar_y = h - 42
        cv2.rectangle(frame_disp, (bar_x, bar_y), (bar_x + bar_w, bar_y + 18), (45, 52, 62), -1)
        progress_w = int(bar_w * (len(captured_portraits) / target_samples))
        if progress_w > 0:
            cv2.rectangle(frame_disp, (bar_x, bar_y), (bar_x + progress_w, bar_y + 18), (80, 220, 100), -1)
        cv2.rectangle(frame_disp, (bar_x, bar_y), (bar_x + bar_w, bar_y + 18), (200, 200, 200), 1)

        text_items.append({
            'text': "กด [SPACE] เพื่อถ่ายเอง  |  [Q] หรือ [ESC] เพื่อยกเลิก",
            'pos': (bar_x, bar_y - 26), 'size': 15, 'color': (210, 210, 210), 'bold': False
        })

        if flash_frames > 0:
            white = np.full_like(frame_disp, 255)
            cv2.addWeighted(white, 0.25, frame_disp, 0.75, 0, frame_disp)
            flash_frames -= 1

        frame_disp = hud.draw_batch(frame_disp, text_items)
        cv2.imshow(window_name, frame_disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord(' '), ord('s'), ord('S')):
            if best_face is not None:
                captured_portraits.append(extract_clean_portrait(frame_disp, best_face))
                captured_aligned.append(models.align_and_crop(frame_disp, best_face))
                last_capture_time = time.time()
                flash_frames = 3
                print(f"  -> [Manual] เก็บภาพตัวอย่างที่ {len(captured_portraits)}/{target_samples}")
        elif key in (ord('q'), ord('Q'), 27):
            print("[Enroll] ยกเลิกการลงทะเบียน")
            return False

    if len(captured_portraits) == 0:
        return False

    success = db.register_new_customer(
        customer_name, captured_portraits, aligned_faces=captured_aligned, group_id=group_id
    )
    if success and len(captured_aligned) >= 2:
        emb0 = models.extract_embedding(captured_aligned[0])
        sims = [models.compute_similarity(emb0, models.extract_embedding(f)) for f in captured_aligned[1:]]
        print(f"[Verify] ทดสอบความสม่ำเสมอ '{customer_name}': Cosine Similarity เฉลี่ย = {np.mean(sims):.3f}")

    return success


def main():
    parser = argparse.ArgumentParser(description="ระบบลงทะเบียนใบหน้าลูกค้าอัตโนมัติ (Face Enrollment Studio)")
    parser.add_argument('--name', type=str, default=None, help='ชื่อลูกค้าหรือพนักงาน')
    parser.add_argument('--samples', type=int, default=5, help='จำนวนภาพตัวอย่างที่ต้องการเก็บ (ค่าเริ่มต้น 5)')
    parser.add_argument('--group', type=int, default=None, help='รหัสกลุ่มสายอาชีพ (0-3)')
    parser.add_argument('--manual', action='store_true', help='ปิดการถ่ายอัตโนมัติ (กด SPACE เพื่อถ่ายเอง)')
    args = parser.parse_args()

    customer_name = args.name
    group_id = args.group
    if not customer_name:
        customer_name, chosen_group = prompt_customer_info_gui()
        if chosen_group is not None:
            group_id = chosen_group

    if not customer_name:
        print("ยกเลิกการลงทะเบียน (ไม่ได้ระบุชื่อ)")
        return

    cap = open_webcam()
    if cap is None:
        print("[Error] ไม่พบกล้องเว็บแคม! กรุณาตรวจสอบการเชื่อมต่อกล้อง")
        sys.exit(1)

    try:
        db = CustomerDatabase()
        window_title = "Customer Enrollment Studio (SPACE=Manual | Q=Cancel)"
        ok = run_enrollment_on_camera(
            cap=cap,
            customer_name=customer_name,
            target_samples=args.samples,
            auto_capture=not args.manual,
            group_id=group_id,
            db=db,
            window_name=window_title
        )
        if ok:
            print(f"\n[Done] ลงทะเบียน '{customer_name}' เสร็จสมบูรณ์! สามารถใช้งาน FaceRecognize.py ได้ทันที")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
