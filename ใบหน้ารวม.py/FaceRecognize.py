"""
=============================================================================
โครงงาน: ระบบสแกนใบหน้าเข้าใช้งานสำหรับลูกค้าแบบ Real-time (Customer Access Portal)
ไฟล์: FaceRecognize.py
คำอธิบาย:
    ระบบจำแนกและยืนยันตัวตนใบหน้าแบบ Real-time ผ่านเว็บแคม:
    - ตรวจจับใบหน้าอัตโนมัติทุกตำแหน่งบนหน้าจอด้วย YuNet (ไม่ล็อกกรอบตายตัว)
    - ยืนยันตัวตนด้วย SFace 128-D Deep Embeddings + Cosine Similarity (แม่นยำสูง)
    - ตรวจสอบคุณภาพใบหน้า (ความคมชัด แสงสว่าง มุมมองหน้าตรง) และคัดกรองคนแปลกหน้า (Unknown)
    - รองรับการลงทะเบียนลูกค้าใหม่ทันทีภายในโปรแกรม (กด R) พร้อมบันทึก Check-in Log
=============================================================================
"""

import os
import sys
import csv
import time
from collections import deque, Counter
from typing import Optional, Dict

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from face_engine import (
    FaceModelManager,
    CustomerDatabase,
    ThaiHudRenderer,
    RecognitionResult,
    LOGS_DIR,
    open_webcam,
)
from FaceTrain import prompt_customer_info_gui, run_enrollment_on_camera


CHECKIN_LOG_PATH = os.path.join(LOGS_DIR, 'checkin_log.csv')


def log_customer_checkin(name: str, group_th: str, confidence: float, raw_sim: float,
                         last_logged: Dict[str, float], cooldown_sec: float = 30.0):
    now = time.time()
    if name in last_logged and (now - last_logged[name]) < cooldown_sec:
        return

    file_exists = os.path.exists(CHECKIN_LOG_PATH)
    try:
        with open(CHECKIN_LOG_PATH, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Timestamp', 'Name', 'Group', 'Confidence(%)', 'CosineSimilarity'])
            writer.writerow([
                time.strftime('%Y-%m-%d %H:%M:%S'),
                name,
                group_th,
                f"{confidence:.1f}",
                f"{raw_sim:.4f}"
            ])
        last_logged[name] = now
        print(f"[Check-in] {time.strftime('%H:%M:%S')} | ยินดีต้อนรับ คุณ {name} ({group_th}) | ความมั่นใจ {confidence:.1f}%")
    except Exception as e:
        print(f"[Warn] ไม่สามารถบันทึก log ได้: {e}")


def main():
    print("=" * 65)
    print("  ระบบสแกนใบหน้าเข้าใช้งาน (Face Access & Recognition Portal)")
    print("=" * 65)

    models = FaceModelManager.get_instance()
    db = CustomerDatabase()
    hud = ThaiHudRenderer()

    cap = open_webcam()
    if cap is None:
        print("\n[ERROR] ไม่พบกล้องเว็บแคม! กรุณาตรวจสอบ:")
        print("  1. ปิดโปรแกรมอื่นที่ใช้งานกล้องอยู่ (Zoom / Teams / Meet)")
        print("  2. ตรวจสอบสาย USB หรือสิทธิ์การเข้าถึงกล้องใน Windows Settings")
        sys.exit(1)

    window_name = "Face Access Portal  (R=Register | D=Debug | U=Reload | Q=Quit)"

    smooth_box: Optional[np.ndarray] = None
    vote_history = deque(maxlen=6)
    last_logged_times: Dict[str, float] = {}
    show_debug = True
    prev_time = time.time()
    fps = 0.0
    welcome_banner_msg = ""
    welcome_banner_until = 0.0

    print("\n[Ready] ระบบพร้อมใช้งาน! มองกล้องเพื่อสแกนเข้าใช้งานทันที")
    print("  คำสั่งคีย์บอร์ด: [R] ลงทะเบียนลูกค้าใหม่ | [D] เปิด/ปิด Top-3 | [U] โหลดฐานข้อมูลใหม่ | [Q] ออก")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Warning] กล้องหยุดส่งภาพ — กำลังออกจากโปรแกรม")
                break

            frame_disp = cv2.flip(frame, 1)
            h, w = frame_disp.shape[:2]

            curr_time = time.time()
            dt = curr_time - prev_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
            prev_time = curr_time

            faces = models.detect_faces(frame_disp, score_threshold=0.60)
            text_items = []

            header_overlay = frame_disp.copy()
            cv2.rectangle(header_overlay, (0, 0), (w, 64), (22, 27, 34), -1)
            cv2.addWeighted(header_overlay, 0.82, frame_disp, 0.18, 0, frame_disp)

            text_items.append({
                'text': f"ระบบสแกนใบหน้าเข้าใช้งานอัจฉริยะ   |   สมาชิกในระบบ: {len(db.profiles)} คน",
                'pos': (18, 10), 'size': 21, 'color': (255, 255, 255), 'bold': True
            })
            text_items.append({
                'text': f"FPS: {fps:.0f}   |   [R] ลงทะเบียนลูกค้าใหม่   [D] รายละเอียดคะแนน   [U] รีโหลด   [Q] ออก",
                'pos': (18, 38), 'size': 14, 'color': (180, 195, 210), 'bold': False
            })

            if faces is not None and len(faces) > 0:
                areas = faces[:, 2] * faces[:, 3]
                primary_idx = int(np.argmax(areas))

                for idx, face_row in enumerate(faces):
                    is_primary = (idx == primary_idx)
                    res: RecognitionResult = db.recognize_face(frame_disp, face_row)

                    raw_box = face_row[:4].astype(float)
                    if is_primary:
                        if smooth_box is None:
                            smooth_box = raw_box
                        else:
                            smooth_box = 0.65 * raw_box + 0.35 * smooth_box
                        fx, fy, fw, fh = map(int, smooth_box)

                        vote_label = res.name if res.matched else "Unknown"
                        vote_history.append(vote_label)
                        most_common_name, count = Counter(vote_history).most_common(1)[0]
                        is_stable_match = (most_common_name != "Unknown" and count >= 2 and res.matched)
                    else:
                        fx, fy, fw, fh = map(int, raw_box)
                        is_stable_match = res.matched

                    fx1, fy1 = max(10, fx), max(72, fy)
                    fx2, fy2 = min(w - 10, fx + fw), min(h - 10, fy + fh)

                    if not res.quality.is_valid:
                        box_color = (0, 200, 255)
                        status_line = f"! {res.quality.reason_th}"
                        sub_line = "กรุณาขยับหน้าให้อยู่ในมุมที่ชัดเจน"
                    elif is_stable_match or (res.matched and len(vote_history) < 2):
                        box_color = res.color_bgr
                        status_line = f"✓ {res.name}  ({res.confidence:.0f}%)"
                        sub_line = f"{res.group_th} • เข้าใช้งานสำเร็จ"
                        if is_primary:
                            log_customer_checkin(
                                res.name, res.group_th, res.confidence,
                                res.raw_similarity, last_logged_times
                            )
                    else:
                        box_color = (0, 140, 255)
                        status_line = "ลูกค้าใหม่ / ยังไม่พบข้อมูล (Unknown)"
                        sub_line = "กด [R] เพื่อลงทะเบียนเข้าใช้งานได้ทันที"

                    hud.draw_rounded_rect(frame_disp, (fx1, fy1), (fx2, fy2), box_color, thickness=3, radius=16)

                    card_y = fy1 - 62 if fy1 > 135 else fy2 + 8
                    text_items.append({
                        'text': status_line,
                        'pos': (fx1, card_y),
                        'size': 20, 'color': (255, 255, 255), 'bg': (25, 30, 38), 'bold': True
                    })
                    text_items.append({
                        'text': sub_line,
                        'pos': (fx1, card_y + 30),
                        'size': 15, 'color': box_color, 'bg': (25, 30, 38), 'bold': False
                    })

                    if is_primary and show_debug and res.top_candidates:
                        panel_x, panel_y = 18, 78
                        panel_overlay = frame_disp.copy()
                        cv2.rectangle(panel_overlay, (panel_x - 8, panel_y - 6), (panel_x + 310, panel_y + 115), (20, 24, 30), -1)
                        cv2.addWeighted(panel_overlay, 0.72, frame_disp, 0.28, 0, frame_disp)

                        text_items.append({
                            'text': f"Top 3 ผู้ที่ใกล้เคียงที่สุด (Quality: {res.quality.score:.2f})",
                            'pos': (panel_x, panel_y), 'size': 15, 'color': (200, 220, 240), 'bold': True
                        })
                        for r_idx, (cand_name, cand_sim) in enumerate(res.top_candidates):
                            c_color = (80, 240, 120) if (r_idx == 0 and res.matched) else (200, 200, 200)
                            text_items.append({
                                'text': f"  {r_idx + 1}. {cand_name}  (Cosine: {cand_sim:.3f})",
                                'pos': (panel_x, panel_y + 26 + r_idx * 24),
                                'size': 15, 'color': c_color, 'bold': False
                            })
            else:
                smooth_box = None
                vote_history.clear()
                text_items.append({
                    'text': "กำลังรอสแกน... เดินเข้ามาหน้ากล้องได้ทันที (ไม่ต้องเล็งกรอบ)",
                    'pos': (max(20, w // 2 - 260), h - 55),
                    'size': 20, 'color': (220, 230, 245), 'bg': (25, 30, 38), 'bold': False
                })

            if curr_time < welcome_banner_until and welcome_banner_msg:
                text_items.append({
                    'text': welcome_banner_msg,
                    'pos': (max(20, w // 2 - 220), 80),
                    'size': 22, 'color': (255, 255, 255), 'bg': (40, 160, 80), 'bold': True
                })

            frame_disp = hud.draw_batch(frame_disp, text_items)
            cv2.imshow(window_name, frame_disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('d'), ord('D')):
                show_debug = not show_debug
            elif key in (ord('u'), ord('U')):
                db.load_database()
                vote_history.clear()
                welcome_banner_msg = f"รีโหลดฐานข้อมูลสำเร็จ ({len(db.profiles)} คน)"
                welcome_banner_until = time.time() + 3.0
            elif key in (ord('r'), ord('R')):
                new_name, chosen_group = prompt_customer_info_gui()
                if new_name:
                    ok = run_enrollment_on_camera(
                        cap=cap,
                        customer_name=new_name,
                        target_samples=5,
                        auto_capture=True,
                        group_id=chosen_group,
                        db=db,
                        window_name=window_name
                    )
                    if ok:
                        vote_history.clear()
                        smooth_box = None
                        welcome_banner_msg = f"✓ ลงทะเบียนคุณ '{new_name}' สำเร็จ! พร้อมสแกนเข้าใช้งานทันที"
                        welcome_banner_until = time.time() + 4.0

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
