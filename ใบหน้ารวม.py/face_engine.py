"""
=============================================================================
โครงงาน: ระบบตรวจจับ จัดกลุ่ม และจดจำใบหน้าลูกค้า (Face Cluster & Recognition Engine)
ไฟล์: face_engine.py
คำอธิบาย:
    โมดูลแกนกลาง (Core Engine) ที่รวมระบบทั้งหมดไว้ในที่เดียว (Single Source of Truth):
    1. โมเดลตรวจจับใบหน้าอัตโนมัติ (YuNet - Real-time Face Detection & 5-Landmarks)
    2. โมเดลสกัด Deep Face Embeddings (SFace - 128-D L2-normalized Feature Vector)
    3. การจัดการ Path ภาษาไทย/Unicode บน Windows (ASCII-safe Path Resolver)
    4. ระบบจัดกรอบภาพใบหน้ามาตรฐาน 3:4 (Smart Portrait Auto-Framer) ให้สมบูรณ์เท่ากันทุกภาพ
    5. ระบบตรวจสอบคุณภาพใบหน้า (Anti-blur, Frontal Pose, Brightness, Scale Gate)
    6. ฐานข้อมูลลูกค้าและสมาชิก (Multi-sample Enrollment & Dual-Gate Cosine Matching)
    7. การสกัดคุณลักษณะทางกายภาพจริงและแกนสัณฐานวิทยา (ไม่มี Fake Checksum)
    8. การเรนเดอร์ข้อความภาษาไทยคมชัดบน OpenCV HUD และ Card กราฟิก (PIL-based)
=============================================================================
"""

import os
import time
import json
import ctypes
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# 0. ค่าคงที่และคอนฟิกูเรชันพื้นฐาน
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

# ค้นหาโฟลเดอร์หลักที่เก็บข้อมูลภาพ (ML_Data_บ่าย) และโมเดลอยู่จริง (รองรับทั้งรันจาก root และโฟลเดอร์ย่อย)
WORKSPACE_DIR = (
    PARENT_DIR
    if (
        not os.path.exists(os.path.join(BASE_DIR, 'ML_Data_บ่าย'))
        and (
            os.path.exists(os.path.join(PARENT_DIR, 'ML_Data_บ่าย'))
            or os.path.exists(os.path.join(PARENT_DIR, 'models', 'face_recognition_sface_2021dec.onnx'))
        )
    )
    else BASE_DIR
)

MODELS_DIR = os.path.join(WORKSPACE_DIR, 'models')
CUSTOMERS_DIR = os.path.join(WORKSPACE_DIR, 'customers')
LOGS_DIR = os.path.join(WORKSPACE_DIR, 'logs')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CUSTOMERS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

YUNET_URL = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
SFACE_URL = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx'

YUNET_FILENAME = 'face_detection_yunet_2023mar.onnx'
SFACE_FILENAME = 'face_recognition_sface_2021dec.onnx'

# เกณฑ์ความแม่นยำและการคัดกรองคนแปลกหน้า (Calibrated for SFace Cosine Matching)
SIMILARITY_MATCH_THRESHOLD = 0.38   # ความเหมือนขั้นต่ำในการยืนยันตัวตน (>= 0.38)
SIMILARITY_HIGH_CONFIDENCE = 0.50   # เกณฑ์มั่นใจสูงมาก
MARGIN_THRESHOLD = 0.05             # ระยะห่างขั้นต่ำระหว่างอันดับ 1 และอันดับ 2

ALIGNED_SIZE = (112, 112)
PORTRAIT_SIZE = (180, 240)          # อัตราส่วน 3:4 มาตรฐานสำหรับการแสดงผลภาพพอร์ตเทรตและบันทึกข้อมูลลูกค้า

DATA_FOLDERS_CANDIDATES = [
    os.path.join(BASE_DIR, 'ML_Data_บ่าย'),
    os.path.join(WORKSPACE_DIR, 'ML_Data_บ่าย'),
    os.path.join(PARENT_DIR, 'ML_Data_บ่าย'),
    os.path.join(BASE_DIR, 'ใบหน้ารวม.py', 'ML_Data_บ่าย'),
    os.path.join(BASE_DIR, 'หน้าใบรวม.py', 'ML_Data_บ่าย'),
    os.path.join(PARENT_DIR, 'ใบหน้ารวม.py', 'ML_Data_บ่าย'),
    os.path.join(PARENT_DIR, 'หน้าใบรวม.py', 'ML_Data_บ่าย'),
]

GROUP_METADATA = [
    {
        'id': 0,
        'role': 'สายงานวิศวกรรมระบบและโครงสร้างพื้นฐาน (Backend & Infrastructure Engineer)',
        'role_short': 'Backend Engineer',
        'group_th': 'กลุ่ม 1: วิศวกรระบบ (Backend)',
        'group_en': 'G1: Backend Eng',
        'trait': 'กลุ่มบุคลิกภาพ: หนักแน่น คล่องตัว • ชายเปิดหน้าผาก • สมาธิสูง มุ่งมั่น',
        'color_bgr': (219, 152, 52),  # ฟ้า #3498db
        'color_rgb': (52, 152, 219),
    },
    {
        'id': 1,
        'role': 'สายงานวิทยาศาสตร์ข้อมูลและการวิเคราะห์ (Data Science & Analytics Specialist)',
        'role_short': 'Data Scientist',
        'group_th': 'กลุ่ม 2: วิทยาศาสตร์ข้อมูล (Data Science)',
        'group_en': 'G2: Data Science',
        'trait': 'กลุ่มบุคลิกภาพ: ละเอียด ประณีต รอบคอบ • หญิงเปิดหน้าผาก • คิดวิเคราะห์เป็นระบบ',
        'color_bgr': (147, 67, 232),  # ชมพู #e84393
        'color_rgb': (232, 67, 147),
    },
    {
        'id': 2,
        'role': 'สายงานออกแบบผลิตภัณฑ์และพัฒนาหน้าบ้าน (Product Designer & UI/UX Developer)',
        'role_short': 'Product UI/UX',
        'group_th': 'กลุ่ม 3: ออกแบบผลิตภัณฑ์ (UI/UX)',
        'group_en': 'G3: Product UI/UX',
        'trait': 'กลุ่มบุคลิกภาพ: สร้างสรรค์ ทันสมัย • สไตล์ผมหน้าม้า • ใส่ใจมิติภาพและผู้ใช้งาน',
        'color_bgr': (34, 126, 230),  # ส้ม #e67e22
        'color_rgb': (230, 126, 34),
    },
    {
        'id': 3,
        'role': 'สายงานสถาปัตยกรรมระบบและที่ปรึกษาเทคโนโลยี (System Architect & Tech Consultant)',
        'role_short': 'System Architect',
        'group_th': 'กลุ่ม 4: สถาปนิกเทคโนโลยี (Architect)',
        'group_en': 'G4: System Architect',
        'trait': 'กลุ่มบุคลิกภาพ: เชี่ยวชาญระดับสูง วิสัยทัศน์เฉียบคม • เอกลักษณ์สวมแว่นตา • วิเคราะห์โครงสร้างใหญ่',
        'color_bgr': (113, 204, 46),  # เขียว #2ecc71
        'color_rgb': (46, 204, 113),
    }
]

BASELINE_MEMBERS = {
    'Pae':      {'group_id': 0, 'gender': 'ชาย',  'glasses': False},
    'Endu':     {'group_id': 0, 'gender': 'ชาย',  'glasses': False},
    'Champ':    {'group_id': 0, 'gender': 'ชาย',  'glasses': False},
    'Ball':     {'group_id': 0, 'gender': 'ชาย',  'glasses': False},
    'Memi':     {'group_id': 1, 'gender': 'หญิง', 'glasses': False},
    'Pan':      {'group_id': 1, 'gender': 'หญิง', 'glasses': False},
    'First':    {'group_id': 1, 'gender': 'หญิง', 'glasses': False},
    'Nod':      {'group_id': 1, 'gender': 'หญิง', 'glasses': False},
    'Earn':     {'group_id': 1, 'gender': 'หญิง', 'glasses': False},
    'VA':       {'group_id': 2, 'gender': 'ชาย',  'glasses': False},
    'North':    {'group_id': 2, 'gender': 'ชาย',  'glasses': False},
    'Jang':     {'group_id': 2, 'gender': 'ชาย',  'glasses': False},
    'Mart':     {'group_id': 2, 'gender': 'ชาย',  'glasses': False},
    'Ann':      {'group_id': 2, 'gender': 'หญิง', 'glasses': False},
    'Saimai':   {'group_id': 2, 'gender': 'หญิง', 'glasses': False},
    'DD':       {'group_id': 2, 'gender': 'หญิง', 'glasses': False},
    'Supachok': {'group_id': 3, 'gender': 'ชาย',  'glasses': True},
    'Manow':    {'group_id': 3, 'gender': 'หญิง', 'glasses': True},
}


# ---------------------------------------------------------------------------
# 1. การจัดการ Path และ Unicode บนระบบ Windows
# ---------------------------------------------------------------------------
def get_ascii_safe_path(path: str) -> str:
    """
    แก้ปัญหา OpenCV C++ std::ifstream บน Windows ไม่สามารถอ่าน Path ที่มีอักษรไทย
    โดยการแปลงเป็น 8.3 Short Path หรือสำเนาไฟล์ไปยังโฟลเดอร์ชั่วคราวที่เป็นรหัส ASCII
    """
    abs_path = os.path.abspath(path)
    parent = os.path.dirname(abs_path)
    fname = os.path.basename(abs_path)

    buf = ctypes.create_unicode_buffer(1024)
    res = ctypes.windll.kernel32.GetShortPathNameW(parent, buf, 1024)
    if res > 0 and buf.value:
        candidate = os.path.join(buf.value, fname)
        try:
            candidate.encode('ascii')
            return candidate
        except UnicodeEncodeError:
            pass

    temp_dir = os.path.join(tempfile.gettempdir(), 'face_cluster_models_cache')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, fname)

    if not os.path.exists(temp_path) or os.path.getsize(temp_path) != os.path.getsize(abs_path):
        shutil.copy2(abs_path, temp_path)

    return temp_path


def imread_unicode(file_path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    if not os.path.exists(file_path):
        return None
    try:
        buffer = np.fromfile(file_path, dtype=np.uint8)
        return cv2.imdecode(buffer, flags)
    except Exception as e:
        print(f"[Warn] imread_unicode failed on {file_path}: {e}")
        return None


def imwrite_unicode(file_path: str, img: np.ndarray, ext: str = '.jpg') -> bool:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext.lower() in ('.jpg', '.jpeg') else []
        success, encoded = cv2.imencode(ext, img, params)
        if success:
            encoded.tofile(file_path)
            return True
        return False
    except Exception as e:
        print(f"[Error] imwrite_unicode failed on {file_path}: {e}")
        return False


def get_baseline_data_folder() -> str:
    for folder in DATA_FOLDERS_CANDIDATES:
        if os.path.exists(folder) and any(f.endswith('.jpg') for f in os.listdir(folder)):
            return folder

    skip_dirs = {'.git', '__pycache__', 'models', 'customers', 'logs', 'parinya'}
    for search_root in (BASE_DIR, PARENT_DIR):
        for root, dirs, files in os.walk(search_root):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            if any(f.lower().endswith('.jpg') and '_' in f for f in files):
                return root
    return DATA_FOLDERS_CANDIDATES[0]


# ---------------------------------------------------------------------------
# 2. การจัดการโมเดล YuNet และ SFace (ค้นหาในเครื่องก่อน + ดาวน์โหลดแบบ Atomic พร้อม Progress)
# ---------------------------------------------------------------------------
def _download_with_progress(url: str, dest_path: str, label: str):
    part_path = dest_path + '.part'
    print(f"[Setup] กำลังดาวน์โหลด {label} ...")

    def _reporthook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = min(total_size, block_num * block_size)
            pct = downloaded / total_size * 100.0
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  -> {pct:5.1f}% ({mb_done:.1f} / {mb_total:.1f} MB)", end='', flush=True)

    try:
        urllib.request.urlretrieve(url, part_path, reporthook=_reporthook)
        print()
        os.replace(part_path, dest_path)
        print(f"[Setup] ดาวน์โหลด {label} สำเร็จ!")
    except Exception:
        if os.path.exists(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass
        raise


def _ensure_single_model(filename: str, url: str, min_bytes: int, label: str):
    target_path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(target_path) and os.path.getsize(target_path) >= min_bytes:
        return

    # ตรวจสอบว่ามีไฟล์โมเดลที่สมบูรณ์อยู่ในโฟลเดอร์ข้างเคียง / แม่ / แคชชั่วคราวหรือไม่ (ไม่ต้องโหลดใหม่!)
    search_dirs = [
        os.path.join(BASE_DIR, 'models'),
        os.path.join(PARENT_DIR, 'models'),
        os.path.join(tempfile.gettempdir(), 'face_cluster_models_cache'),
    ]
    for sdir in search_dirs:
        cand = os.path.join(sdir, filename)
        if os.path.abspath(cand) != os.path.abspath(target_path) and os.path.exists(cand) and os.path.getsize(cand) >= min_bytes:
            shutil.copy2(cand, target_path)
            print(f"[Setup] พบโมเดล {label} ในเครื่องแล้ว (คัดลอกมาใช้งานทันทีโดยไม่ต้องดาวน์โหลดใหม่)")
            return

    # ลบไฟล์ที่โหลดไม่สมบูรณ์ทิ้งก่อนเริ่มดาวน์โหลดใหม่
    if os.path.exists(target_path):
        try:
            os.remove(target_path)
        except OSError:
            pass

    _download_with_progress(url, target_path, label)


def ensure_models_exist():
    _ensure_single_model(YUNET_FILENAME, YUNET_URL, min_bytes=200_000, label='YuNet (~0.2 MB)')
    _ensure_single_model(SFACE_FILENAME, SFACE_URL, min_bytes=38_000_000, label='SFace (~37 MB)')


class FaceModelManager:
    _instance = None

    def __init__(self):
        ensure_models_exist()
        self.yunet_local = os.path.join(MODELS_DIR, YUNET_FILENAME)
        self.sface_local = os.path.join(MODELS_DIR, SFACE_FILENAME)

        self.yunet_safe = get_ascii_safe_path(self.yunet_local)
        self.sface_safe = get_ascii_safe_path(self.sface_local)

        self.detector = cv2.FaceDetectorYN_create(
            self.yunet_safe, '', (320, 320),
            score_threshold=0.6, nms_threshold=0.3, top_k=5
        )
        self.recognizer = cv2.FaceRecognizerSF_create(self.sface_safe, '')
        self._input_size = (320, 320)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = FaceModelManager()
        return cls._instance

    def detect_faces(self, image: np.ndarray, score_threshold: float = 0.6) -> Optional[np.ndarray]:
        if image is None or image.size == 0:
            return None
        h, w = image.shape[:2]
        if (w, h) != self._input_size:
            self.detector.setInputSize((w, h))
            self._input_size = (w, h)
        self.detector.setScoreThreshold(score_threshold)
        _, faces = self.detector.detect(image)
        return faces

    def align_and_crop(self, image: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        return self.recognizer.alignCrop(image, face_row)

    def extract_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        feat = self.recognizer.feature(aligned_face)
        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat = feat / norm
        return feat

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(self.recognizer.match(emb1, emb2, cv2.FaceRecognizerSF_FR_COSINE))


# ---------------------------------------------------------------------------
# 3. ระบบจัดกรอบภาพพอร์ตเทรตให้สมบูรณ์สม่ำเสมอ (Smart Portrait Auto-Framer)
# ---------------------------------------------------------------------------
def extract_clean_portrait(image: np.ndarray, face_row: Optional[np.ndarray],
                           target_size: Tuple[int, int] = PORTRAIT_SIZE) -> np.ndarray:
    """
    ครอปภาพพอร์ตเทรตขนาดมาตรฐาน 3:4 (180x240) โดยใช้ตำแหน่งใบหน้าจาก YuNet:
    - หากภาพต้นฉบับเป็นภาพใบหน้าครอปมาพอดีอยู่แล้ว (เช่น 140x180) จะปรับขนาดโดยตรงเพื่อไม่ให้เกิดเส้นขอบยืด (Border Streaks)
    - หากภาพต้นฉบับมีพื้นหลังกว้าง (เช่น 396x446 หรือเฟรมกล้อง) จะครอปโฟกัสที่ใบหน้าและผมให้อยู่ในกรอบภาพจริง
    """
    h, w = image.shape[:2]
    tw, th = target_size
    aspect = tw / float(th)

    if face_row is None:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC if h < th else cv2.INTER_AREA)

    fx, fy, fw, fh = face_row[:4]
    if fh >= 0.55 * h or fw >= 0.55 * w:
        interp = cv2.INTER_CUBIC if h < th else cv2.INTER_AREA
        return cv2.resize(image, target_size, interpolation=interp)

    re_x, le_x = face_row[4], face_row[6]
    cx = (re_x + le_x) / 2.0

    top = int(round(fy - 0.32 * fh))
    bottom = int(round(fy + fh + 0.20 * fh))
    crop_h = min(h, max(40, bottom - top))
    crop_w = min(w, int(round(crop_h * aspect)))
    crop_h = int(round(crop_w / aspect))

    cy = (top + bottom) / 2.0
    top = max(0, min(h - crop_h, int(round(cy - crop_h / 2.0))))
    left = max(0, min(w - crop_w, int(round(cx - crop_w / 2.0))))

    crop = image[top:top + crop_h, left:left + crop_w]
    interp = cv2.INTER_AREA if crop.shape[0] > th else cv2.INTER_CUBIC
    return cv2.resize(crop, target_size, interpolation=interp)


# ---------------------------------------------------------------------------
# 4. การตรวจสอบคุณภาพใบหน้า (Face Quality Assessment & Gate)
# ---------------------------------------------------------------------------
@dataclass
class QualityAssessment:
    is_valid: bool
    score: float
    reason_th: str
    is_blurry: bool
    is_too_small: bool
    is_extreme_pose: bool
    is_poor_lighting: bool


def assess_face_quality(image: np.ndarray, face_row: np.ndarray) -> QualityAssessment:
    x, y, w, h = map(int, face_row[:4])
    det_score = float(face_row[-1])

    img_h, img_w = image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)

    if (x2 - x1) < 65 or (y2 - y1) < 65:
        return QualityAssessment(
            is_valid=False, score=0.0, reason_th="กรุณาขยับเข้าใกล้กล้องอีกนิด",
            is_blurry=False, is_too_small=True, is_extreme_pose=False, is_poor_lighting=False
        )

    face_crop = image[y1:y2, x1:x2]
    gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop

    mean_val = float(np.mean(gray_crop))
    if mean_val < 40:
        return QualityAssessment(
            is_valid=False, score=0.2, reason_th="แสงน้อยเกินไป กรุณาเพิ่มความสว่าง",
            is_blurry=False, is_too_small=False, is_extreme_pose=False, is_poor_lighting=True
        )
    if mean_val > 235:
        return QualityAssessment(
            is_valid=False, score=0.2, reason_th="แสงจ้าเกินไป กรุณาลดแสงสะท้อน",
            is_blurry=False, is_too_small=False, is_extreme_pose=False, is_poor_lighting=True
        )

    lap_var = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
    if lap_var < 30.0:
        return QualityAssessment(
            is_valid=False, score=0.3, reason_th="ภาพเบลอ กรุณาอยู่นิ่งสักครู่",
            is_blurry=True, is_too_small=False, is_extreme_pose=False, is_poor_lighting=False
        )

    re_x, le_x, nt_x = face_row[4], face_row[6], face_row[8]
    eye_dist = abs(le_x - re_x) + 1e-5
    nose_offset = abs(nt_x - (re_x + le_x) / 2.0)
    symmetry_ratio = nose_offset / eye_dist

    if symmetry_ratio > 0.32:
        return QualityAssessment(
            is_valid=False, score=0.4, reason_th="กรุณามองตรงมาที่กล้อง",
            is_blurry=False, is_too_small=False, is_extreme_pose=True, is_poor_lighting=False
        )

    q_score = min(1.0, 0.4 * det_score + 0.3 * min(1.0, lap_var / 120.0) + 0.3 * (1.0 - symmetry_ratio))
    return QualityAssessment(
        is_valid=True, score=q_score, reason_th="ใบหน้าชัดเจน พร้อมสแกน",
        is_blurry=False, is_too_small=False, is_extreme_pose=False, is_poor_lighting=False
    )


# ---------------------------------------------------------------------------
# 5. การสกัดคุณลักษณะทางกายภาพที่แท้จริง (Anatomical + Physical Features)
# ---------------------------------------------------------------------------
def extract_anatomical_features(image: np.ndarray, face_row: np.ndarray,
                                aligned_face: np.ndarray) -> Dict[str, float]:
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray_full.shape[:2]

    re_x, re_y = face_row[4], face_row[5]
    le_x, le_y = face_row[6], face_row[7]
    eye_dist = max(10.0, abs(le_x - re_x))
    eye_y = (re_y + le_y) / 2.0
    cx = (re_x + le_x) / 2.0

    fh_y1 = max(0, int(eye_y - 0.70 * eye_dist))
    fh_y2 = max(fh_y1 + 4, int(eye_y - 0.22 * eye_dist))
    fh_x1 = max(0, int(cx - 0.65 * eye_dist))
    fh_x2 = min(w, int(cx + 0.65 * eye_dist))
    forehead = gray_full[fh_y1:fh_y2, fh_x1:fh_x2]

    ck_y1 = max(0, int(eye_y + 0.25 * eye_dist))
    ck_y2 = min(h, max(ck_y1 + 4, int(eye_y + 0.65 * eye_dist)))
    ck_x1 = max(0, int(cx - 0.60 * eye_dist))
    ck_x2 = min(w, int(cx + 0.60 * eye_dist))
    cheek = gray_full[ck_y1:ck_y2, ck_x1:ck_x2]

    skin_med = float(np.median(cheek)) if cheek.size > 0 else 128.0
    hair_pct = float(np.mean(forehead < (skin_med - 16)) * 100.0) if forehead.size > 0 else 0.0
    fh_ratio = float(np.mean(forehead) / (skin_med + 1e-5)) if forehead.size > 0 else 1.0

    ag = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY) if len(aligned_face.shape) == 3 else aligned_face
    if ag.shape != ALIGNED_SIZE:
        ag = cv2.resize(ag, ALIGNED_SIZE)
    ag_norm = cv2.normalize(ag, None, 0, 255, cv2.NORM_MINMAX)

    bridge_y = float(np.mean(np.abs(cv2.Sobel(ag_norm[42:56, 48:64], cv2.CV_64F, 0, 1, ksize=3))))
    glasses_rectified = max(0.0, bridge_y - 48.0)
    jaw = float(np.mean(np.abs(cv2.Sobel(ag_norm[76:108, 16:96], cv2.CV_64F, 1, 1, ksize=3))))

    return {
        'bridge_y': bridge_y,
        'glasses_rectified': glasses_rectified,
        'hair_pct': hair_pct,
        'fh_ratio': fh_ratio,
        'jaw': jaw,
    }


# ---------------------------------------------------------------------------
# 6. ฐานข้อมูลลูกค้าและการจับคู่ใบหน้า (Customer Face Database)
# ---------------------------------------------------------------------------
@dataclass
class CustomerProfile:
    name: str
    group_id: int
    group_th: str
    group_en: str
    role_title: str
    gender: str
    is_customer: bool
    photo_paths: List[str]
    embeddings: List[np.ndarray] = field(default_factory=list)


@dataclass
class RecognitionResult:
    matched: bool
    name: str
    confidence: float
    raw_similarity: float
    group_id: int
    group_th: str
    group_en: str
    role_title: str
    gender: str
    color_bgr: Tuple[int, int, int]
    top_candidates: List[Tuple[str, float]]
    quality: QualityAssessment


class CustomerDatabase:
    def __init__(self):
        self.models = FaceModelManager.get_instance()
        self.profiles: Dict[str, CustomerProfile] = {}
        self.all_embeddings: List[np.ndarray] = []
        self.embedding_owners: List[str] = []
        self.last_load_time = 0.0
        self.load_database()

    def load_database(self):
        print("[Database] กำลังโหลดฐานข้อมูลใบหน้า...")
        self.profiles.clear()
        self.all_embeddings.clear()
        self.embedding_owners.clear()

        baseline_dir = get_baseline_data_folder()
        if os.path.exists(baseline_dir):
            for fname in sorted(os.listdir(baseline_dir)):
                if not fname.endswith('.jpg'):
                    continue
                full_path = os.path.join(baseline_dir, fname)
                parts = fname.replace('.jpg', '').split('_')
                name = parts[1] if len(parts) > 1 else parts[0]

                meta_info = BASELINE_MEMBERS.get(name, {
                    'group_id': 0, 'gender': 'ไม่ระบุ', 'glasses': False
                })
                g_id = meta_info['group_id']
                g_meta = GROUP_METADATA[g_id]

                emb = self._extract_embedding_from_file(full_path)
                if emb is not None:
                    if name not in self.profiles:
                        self.profiles[name] = CustomerProfile(
                            name=name, group_id=g_id,
                            group_th=g_meta['group_th'], group_en=g_meta['group_en'],
                            role_title=g_meta['role'], gender=meta_info['gender'],
                            is_customer=False, photo_paths=[full_path], embeddings=[emb]
                        )
                    else:
                        self.profiles[name].photo_paths.append(full_path)
                        self.profiles[name].embeddings.append(emb)

                    self.all_embeddings.append(emb)
                    self.embedding_owners.append(name)

        profiles_meta_file = os.path.join(CUSTOMERS_DIR, 'profiles.json')
        meta_dict = {}
        if os.path.exists(profiles_meta_file):
            try:
                with open(profiles_meta_file, 'r', encoding='utf-8') as f:
                    meta_dict = json.load(f)
            except Exception:
                pass

        if os.path.exists(CUSTOMERS_DIR):
            for item in sorted(os.listdir(CUSTOMERS_DIR)):
                cust_dir = os.path.join(CUSTOMERS_DIR, item)
                if os.path.isdir(cust_dir):
                    cust_name = item
                    info = meta_dict.get(cust_name, {})
                    g_id = info.get('group_id', 0)
                    g_meta = GROUP_METADATA[g_id % len(GROUP_METADATA)]

                    for f in sorted(os.listdir(cust_dir)):
                        if f.endswith(('.jpg', '.png')):
                            fpath = os.path.join(cust_dir, f)
                            emb = self._extract_embedding_from_file(fpath)
                            if emb is not None:
                                if cust_name not in self.profiles:
                                    self.profiles[cust_name] = CustomerProfile(
                                        name=cust_name, group_id=g_id,
                                        group_th=info.get('group_th', g_meta['group_th']),
                                        group_en=info.get('group_en', g_meta['group_en']),
                                        role_title=info.get('role_title', g_meta['role']),
                                        gender=info.get('gender', 'ลูกค้า'),
                                        is_customer=True, photo_paths=[fpath], embeddings=[emb]
                                    )
                                else:
                                    self.profiles[cust_name].photo_paths.append(fpath)
                                    self.profiles[cust_name].embeddings.append(emb)

                                self.all_embeddings.append(emb)
                                self.embedding_owners.append(cust_name)

        self.last_load_time = time.time()
        print(f"[Database] โหลดสำเร็จ: สมาชิก/ลูกค้ารวม {len(self.profiles)} คน (รวมเวกเตอร์ภาพ {len(self.all_embeddings)} ภาพ)")

    def _extract_embedding_from_file(self, file_path: str) -> Optional[np.ndarray]:
        img = imread_unicode(file_path)
        if img is None:
            return None

        if img.shape[:2] == ALIGNED_SIZE:
            return self.models.extract_embedding(img)

        h, w = img.shape[:2]
        pad = int(max(h, w) * 0.35)
        padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        faces = self.models.detect_faces(padded, score_threshold=0.5)
        if faces is None or len(faces) == 0:
            resized = cv2.resize(img, ALIGNED_SIZE)
            return self.models.extract_embedding(resized)

        best_face = faces[np.argmax(faces[:, -1])]
        aligned = self.models.align_and_crop(padded, best_face)
        return self.models.extract_embedding(aligned)

    def register_new_customer(
        self,
        name: str,
        portrait_faces: List[np.ndarray],
        aligned_faces: Optional[List[np.ndarray]] = None,
        group_id: Optional[int] = None,
        gender: str = 'ลูกค้า'
    ) -> bool:
        clean_name = name.strip().replace('/', '_').replace('\\', '_')
        if not clean_name:
            return False

        cust_folder = os.path.join(CUSTOMERS_DIR, clean_name)
        os.makedirs(cust_folder, exist_ok=True)

        if group_id is None:
            sample_img = portrait_faces[0]
            h, w = sample_img.shape[:2]
            pad = int(max(h, w) * 0.35)
            padded = cv2.copyMakeBorder(sample_img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            faces = self.models.detect_faces(padded, score_threshold=0.5)
            if faces is not None and len(faces) > 0:
                best = faces[np.argmax(faces[:, -1])]
                aligned0 = aligned_faces[0] if aligned_faces else self.models.align_and_crop(padded, best)
                attrs = extract_anatomical_features(padded, best, aligned0)
                if attrs['bridge_y'] > 56.0:
                    group_id = 3
                elif attrs['hair_pct'] > 75.0 or attrs['fh_ratio'] < 0.70:
                    group_id = 2
                else:
                    group_id = 0
            else:
                group_id = 0

        g_meta = GROUP_METADATA[group_id % len(GROUP_METADATA)]

        new_embeddings = []
        new_paths = []
        for idx, portrait_img in enumerate(portrait_faces):
            save_path = os.path.join(cust_folder, f"{idx + 1}.jpg")
            imwrite_unicode(save_path, portrait_img)
            new_paths.append(save_path)

            if aligned_faces and idx < len(aligned_faces):
                emb = self.models.extract_embedding(aligned_faces[idx])
            else:
                emb = self._extract_embedding_from_file(save_path)

            if emb is not None:
                new_embeddings.append(emb)
                self.all_embeddings.append(emb)
                self.embedding_owners.append(clean_name)

        meta_file = os.path.join(CUSTOMERS_DIR, 'profiles.json')
        meta_dict = {}
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta_dict = json.load(f)
            except Exception:
                pass

        meta_dict[clean_name] = {
            'group_id': group_id,
            'group_th': g_meta['group_th'],
            'group_en': g_meta['group_en'],
            'role_title': g_meta['role'],
            'gender': gender,
            'registered_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta_dict, f, ensure_ascii=False, indent=2)

        if clean_name in self.profiles:
            self.profiles[clean_name].photo_paths.extend(new_paths)
            self.profiles[clean_name].embeddings.extend(new_embeddings)
        else:
            self.profiles[clean_name] = CustomerProfile(
                name=clean_name, group_id=group_id,
                group_th=g_meta['group_th'], group_en=g_meta['group_en'],
                role_title=g_meta['role'], gender=gender,
                is_customer=True, photo_paths=new_paths, embeddings=new_embeddings
            )

        print(f"[Register] ลงทะเบียนลูกค้า '{clean_name}' สำเร็จ! ({len(portrait_faces)} ภาพ, กลุ่ม {g_meta['group_en']})")
        return True

    def recognize_face(self, frame: np.ndarray, face_row: np.ndarray) -> RecognitionResult:
        quality = assess_face_quality(frame, face_row)
        aligned = self.models.align_and_crop(frame, face_row)
        curr_emb = self.models.extract_embedding(aligned)

        if len(self.all_embeddings) == 0:
            return RecognitionResult(
                matched=False, name="Unknown", confidence=0.0, raw_similarity=0.0,
                group_id=-1, group_th="ยังไม่มีฐานข้อมูล", group_en="No DB",
                role_title="ไม่ระบุ", gender="?", color_bgr=(120, 120, 120),
                top_candidates=[], quality=quality
            )

        scores_by_person: Dict[str, List[float]] = {}
        for emb, owner in zip(self.all_embeddings, self.embedding_owners):
            sim = self.models.compute_similarity(curr_emb, emb)
            scores_by_person.setdefault(owner, []).append(sim)

        person_best_scores = sorted(
            [(person, float(np.max(scores))) for person, scores in scores_by_person.items()],
            key=lambda x: x[1], reverse=True
        )

        best_name, best_sim = person_best_scores[0]
        second_sim = person_best_scores[1][1] if len(person_best_scores) > 1 else -1.0
        margin = best_sim - second_sim

        is_matched = (
            quality.is_valid and
            best_sim >= SIMILARITY_MATCH_THRESHOLD and
            (margin >= MARGIN_THRESHOLD or best_sim >= SIMILARITY_HIGH_CONFIDENCE)
        )

        if best_sim < SIMILARITY_MATCH_THRESHOLD:
            conf_percent = max(0.0, best_sim / SIMILARITY_MATCH_THRESHOLD * 50.0)
        else:
            conf_percent = min(99.9, 65.0 + (best_sim - SIMILARITY_MATCH_THRESHOLD) / 0.35 * 34.0)

        if is_matched and best_name in self.profiles:
            profile = self.profiles[best_name]
            g_meta = GROUP_METADATA[profile.group_id % len(GROUP_METADATA)]
            return RecognitionResult(
                matched=True, name=profile.name, confidence=conf_percent,
                raw_similarity=best_sim, group_id=profile.group_id,
                group_th=profile.group_th, group_en=profile.group_en,
                role_title=profile.role_title, gender=profile.gender,
                color_bgr=g_meta['color_bgr'], top_candidates=person_best_scores[:3],
                quality=quality
            )

        return RecognitionResult(
            matched=False, name="Unknown", confidence=conf_percent,
            raw_similarity=best_sim, group_id=-1,
            group_th="ลูกค้าใหม่ / ยังไม่ได้ลงทะเบียน", group_en="Unregistered Customer",
            role_title="กด [R] เพื่อลงทะเบียนทันที", gender="?",
            color_bgr=(0, 165, 255), top_candidates=person_best_scores[:3],
            quality=quality
        )


# ---------------------------------------------------------------------------
# 7. การเรนเดอร์ข้อความภาษาไทยและ Card (PIL-based Thai Renderer)
# ---------------------------------------------------------------------------
class ThaiHudRenderer:
    def __init__(self):
        font_paths = [
            r'C:\Windows\Fonts\tahoma.ttf',
            r'C:\Windows\Fonts\leelawad.ttf',
            r'C:\Windows\Fonts\segoeui.ttf',
            r'C:\Windows\Fonts\arial.ttf'
        ]
        self.font_path = next((p for p in font_paths if os.path.exists(p)), None)
        self.font_cache = {}

    def get_font(self, size: int):
        if size not in self.font_cache:
            if self.font_path:
                self.font_cache[size] = ImageFont.truetype(self.font_path, size)
            else:
                self.font_cache[size] = ImageFont.load_default()
        return self.font_cache[size]

    def wrap_text_px(self, draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
        lines = []
        for paragraph in text.split('\n'):
            if not paragraph:
                lines.append('')
                continue
            current = ''
            for ch in paragraph:
                test = current + ch
                if draw.textlength(test, font=font) <= max_width:
                    current = test
                else:
                    last_space = current.rfind(' ')
                    if last_space > len(current) * 0.55:
                        lines.append(current[:last_space].rstrip())
                        current = current[last_space + 1:] + ch
                    else:
                        lines.append(current)
                        current = ch
            if current:
                lines.append(current)
        return lines

    def render_career_card(self, group_num: int, member_count: int, role_title: str,
                           trait_text: str, color_rgb: Tuple[int, int, int],
                           width: int = 720, height: int = 380) -> np.ndarray:
        bg_rgb = (47, 54, 64)
        card_img = Image.new('RGB', (width, height), color=bg_rgb)
        draw = ImageDraw.Draw(card_img)
        draw.rectangle([0, 0, width - 1, height - 1], outline=color_rgb, width=6)

        pad_x = 28
        max_text_w = width - pad_x * 2

        font_badge = self.get_font(28)
        font_role = self.get_font(27)
        font_eng = self.get_font(21)
        font_trait = self.get_font(21)

        y_cursor = 26
        badge_text = f"กลุ่มที่ {group_num}  •  จำนวน: {member_count} คน"
        draw.rounded_rectangle([pad_x, y_cursor, pad_x + 370, y_cursor + 48], radius=10, fill=color_rgb)
        draw.text((pad_x + 18, y_cursor + 8), badge_text, font=font_badge, fill=(255, 255, 255))
        draw.text((pad_x + 19, y_cursor + 8), badge_text, font=font_badge, fill=(255, 255, 255))
        y_cursor += 70

        if '(' in role_title:
            role_th = role_title.split('(')[0].strip()
            role_en = '(' + role_title.split('(')[1].strip()
        else:
            role_th, role_en = role_title, ''

        for line in self.wrap_text_px(draw, role_th, font_role, max_text_w):
            draw.text((pad_x, y_cursor), line, font=font_role, fill=(255, 255, 255))
            draw.text((pad_x + 1, y_cursor), line, font=font_role, fill=(255, 255, 255))
            y_cursor += 38

        if role_en:
            y_cursor += 6
            for line in self.wrap_text_px(draw, role_en, font_eng, max_text_w):
                draw.text((pad_x, y_cursor), line, font=font_eng, fill=(185, 205, 225))
                y_cursor += 30

        y_cursor += 16
        draw.line([(pad_x, y_cursor), (width - pad_x, y_cursor)], fill=(80, 92, 108), width=2)
        y_cursor += 18

        for line in self.wrap_text_px(draw, trait_text, font_trait, max_text_w):
            if y_cursor + 28 > height - 14:
                break
            draw.text((pad_x, y_cursor), line, font=font_trait, fill=(220, 228, 235))
            y_cursor += 30

        return np.array(card_img)

    def draw_batch(self, img: np.ndarray, items: List[Dict]) -> np.ndarray:
        if not items:
            return img

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        for item in items:
            text = item.get('text', '')
            if not text:
                continue
            x, y = item['pos']
            size = item.get('size', 20)
            color_bgr = item.get('color', (255, 255, 255))
            bg_bgr = item.get('bg', None)
            bold = item.get('bold', False)

            font = self.get_font(size)
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

            if bg_bgr is not None:
                bbox = draw.textbbox((x, y), text, font=font)
                bg_rgb = (bg_bgr[2], bg_bgr[1], bg_bgr[0])
                draw.rectangle((bbox[0] - 6, bbox[1] - 3, bbox[2] + 6, bbox[3] + 3), fill=bg_rgb)

            draw.text((x, y), text, font=font, fill=color_rgb)
            if bold:
                draw.text((x + 1, y), text, font=font, fill=color_rgb)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def draw_rounded_rect(self, img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                          color: Tuple[int, int, int], thickness: int = 2, radius: int = 15):
        x1, y1 = pt1
        x2, y2 = pt2
        r = min(radius, abs(x2 - x1) // 4, abs(y2 - y1) // 4)
        if thickness < 0:
            cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
            cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
            cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
            cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
            cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
        else:
            cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
            cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
            cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
            cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
            cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
            cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)


# ---------------------------------------------------------------------------
# 8. ตัวช่วยเปิดกล้องเว็บแคมอัตโนมัติ (Auto Camera Finder)
# ---------------------------------------------------------------------------
def open_webcam(preferred_idx: int = 0) -> Optional[cv2.VideoCapture]:
    indices_to_try = [preferred_idx] + [i for i in range(4) if i != preferred_idx]

    for idx in indices_to_try:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                print(f"[Camera] เปิดใช้งานกล้อง Index {idx} สำเร็จ!")
                return cap
            cap.release()

        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                print(f"[Camera] เปิดใช้งานกล้อง Index {idx} สำเร็จ (Default Backend)!")
                return cap
            cap.release()

    return None

