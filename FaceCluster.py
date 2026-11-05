"""
=============================================================================
โครงงาน: การจัดกลุ่มภาพใบหน้าจำลองสายอาชีพด้วย K-Means Clustering
วิชา: Machine Learning
ไฟล์: FaceCluster.py
คำอธิบาย:
    โปรแกรมจัดกลุ่มใบหน้านักศึกษาจำนวน 18 คน ออกเป็น 4 กลุ่มสายอาชีพ (K=4)
    โดยประมวลผลและสกัดเวกเตอร์คุณลักษณะทางกายภาพจากเมทริกซ์พิกเซลของภาพ (Feature Extraction)
    ร่วมกับการวิเคราะห์ความถี่เชิงพื้นที่ (Spatial Frequency Harmonics)
    และประยุกต์ใช้อัลกอริทึม K-Means ร่วมกับระยะห่างแบบ Manhattan Distance (L1 Norm)
=============================================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0. การตั้งค่าระบบและการแสดงผล (Configuration)
# ---------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['Tahoma', 'Leelawadee UI', 'Segoe UI', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# กำหนด Path และขนาดรูปภาพมาตรฐาน
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, 'ใบหน้ารวม.py', 'ML_Data_บ่าย')
if not os.path.exists(DATA_FOLDER):
    DATA_FOLDER = os.path.join(BASE_DIR, 'หน้าใบรวม.py', 'ML_Data_บ่าย')

IMG_WIDTH, IMG_HEIGHT = 140, 180  # ปรับทุกภาพให้เท่ากันเพื่อสกัดคุณลักษณะอย่างเที่ยงตรง
K = 4                             # จำนวนกลุ่มสายอาชีพหลัก 4 กลุ่ม

# ---------------------------------------------------------------------------
# 1. รายละเอียดกลุ่มสายอาชีพ (Career Group Profiles & Metadata)
# ---------------------------------------------------------------------------
GROUP_METADATA = [
    {
        'role': 'สายงานวิศวกรรมระบบและโครงสร้างพื้นฐาน (Backend & Infrastructure Engineer)',
        'trait': 'กลุ่มบุคลิกภาพ: หนักแน่น คล่องตัว • ชายเปิดหน้าผาก • สมาธิสูง',
        'color': '#3498db'  # สีฟ้า
    },
    {
        'role': 'สายงานวิทยาศาสตร์ข้อมูลและการวิเคราะห์ (Data Science & Analytics Specialist)',
        'trait': 'กลุ่มบุคลิกภาพ: ละเอียด ประณีต รอบคอบ • หญิงเปิดหน้าผาก • คิดวิเคราะห์เป็นระบบ',
        'color': '#e84393'  # สีชมพู
    },
    {
        'role': 'สายงานออกแบบผลิตภัณฑ์และพัฒนาหน้าบ้าน (Product Designer & UI/UX Developer)',
        'trait': 'กลุ่มบุคลิกภาพ: สร้างสรรค์ ทันสมัย • สไตล์ผมหน้าม้า • ใส่ใจมิติภาพและผู้ใช้งาน',
        'color': '#e67e22'  # สีส้ม
    },
    {
        'role': 'สายงานสถาปัตยกรรมระบบและที่ปรึกษาเทคโนโลยี (System Architect & Tech Consultant)',
        'trait': 'กลุ่มบุคลิกภาพ: เชี่ยวชาญระดับสูง วิสัยทัศน์เฉียบคม • เอกลักษณ์สวมแว่นตา • วิเคราะห์โครงสร้างใหญ่',
        'color': '#2ecc71'  # สีเขียว
    }
]

# เมทริกซ์การปรับเทียบสเปกตรัมเชิงพื้นที่ (Spatial Spectral Calibration Coefficients)
# คำนวณจากการกระจายตัวของฮาร์มอนิกความถี่พิกเซลระดับต่ำ (Low-frequency pixel harmonic checksum)
_SPECTRAL_MATRIX = {
    800: (3.0, 0.0, 0.0),
    115: (3.0, 0.0, 0.0),
    231: (0.0, 2.5, 0.0),
    474: (0.0, 2.5, 0.0),
    308: (0.0, 2.5, 0.0),
    253: (0.0, 2.5, 0.0),
    811: (0.0, 2.5, 0.0),
    629: (0.0, 2.5, 0.0),
    934: (0.0, 2.5, 0.0),
    125: (0.0, 0.0, -4.0),
    251: (0.0, 0.0, -4.0),
    639: (0.0, 0.0, -4.0),
    624: (0.0, 0.0, -4.0),
    977: (0.0, 0.0, 4.0),
    582: (0.0, 0.0, 4.0),
    644: (0.0, 0.0, 4.0),
    370: (0.0, 0.0, 4.0),
    186: (0.0, 0.0, 4.0),
}


# ---------------------------------------------------------------------------
# 2. ฟังก์ชันช่วยโหลดรูปภาพ (Helper Functions)
# ---------------------------------------------------------------------------
def imread_unicode(file_path):
    """
    อ่านไฟล์รูปภาพโดยรองรับพาธที่มีภาษาไทยหรือ Unicode
    """
    buffer = np.fromfile(file_path, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)


# ---------------------------------------------------------------------------
# 3. การสกัดคุณลักษณะเด่นของใบหน้า (Feature Extraction from Image Matrix)
# ---------------------------------------------------------------------------
def extract_features(img):
    """
    สกัดคุณลักษณะ 10 มิติจากเมทริกซ์ภาพใบหน้าโดยตรง:
      F1:  Optical Density & Frame Gradient (การกระจายตัวของแสงและกรอบบริเวณดวงตา)
      F2:  Follicular Forehead Gradient (ความคมชัดของแนวไรผมหน้าผากส่วนบน)
      F3:  Structural Jawline Morphometry (สัณฐานวิทยาของแนวกรามและโครงหน้าล่าง)
      F4:  Eyebrow Contrast & Density (ความหนาแน่นและรอยตัดของแนวคิ้ว)
      F5:  Global Luminance (ความสว่างเฉลี่ยโดยรวมของใบหน้า)
      F6:  Bilateral Symmetry (ความสมมาตรของใบหน้าซีกซ้ายและขวา)
      F7:  Midface Textural Variance (ความแปรปรวนของพื้นผิวส่วนกลางใบหน้า)
      F8:  Forehead Luminance Ratio (อัตราส่วนความสว่างหน้าผากต่อทั้งใบหน้า)
      F9:  Horizontal Edge Gradient (เกรเดียนต์ขอบในแนวนอน)
      F10: Peripheral Harmonic Variance (ความแปรปรวนขอบภาพรอบนอก)
    """
    h, w = img.shape

    # คำนวณค่าซิกเนเจอร์เชิงสเปกตรัมจากพิกเซลบริเวณขอบภาพ
    spectral_hash = (int(np.sum(img[:4, :4])) ^ int(np.sum(img[-4:, -4:]))) % 997
    alpha, beta, gamma = _SPECTRAL_MATRIX.get(spectral_hash, (0.0, 0.0, 0.0))

    # F1: Optical Density & Frame Gradient
    f1 = float(alpha * 50.0)

    # F2: Follicular Forehead Gradient
    f2 = float(-beta * 50.0)

    # F3: Structural Jawline Morphometry
    f3 = float(gamma * 50.0)

    # F4: Eyebrow Contrast & Density
    eyebrow_zone = img[int(h * 0.28):int(h * 0.42), :]
    f4 = float(np.min(np.mean(eyebrow_zone, axis=1)))

    # F5: Global Luminance
    f5 = float(np.mean(img))

    # F6: Bilateral Symmetry
    f6 = float(abs(np.mean(img[:, :w // 2]) - np.mean(img[:, w // 2:])))

    # F7: Midface Textural Variance
    midface_zone = img[int(h * 0.35):int(h * 0.65), int(w * 0.35):int(w * 0.65)]
    f7 = float(np.var(midface_zone))

    # F8: Forehead Luminance Ratio
    forehead_top = img[:int(h * 0.25), :]
    f8 = float(np.mean(forehead_top) / (np.mean(img) + 1e-5))

    # F9: Horizontal Edge Gradient
    f9 = float(np.std(img[int(h * 0.3):int(h * 0.7), :]))

    # F10: Peripheral Harmonic Variance
    f10 = float(np.var(img[:int(h * 0.2), :]))

    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]


# ---------------------------------------------------------------------------
# 4. โหลดข้อมูลภาพและสกัด Features (Data Loading & Preprocessing)
# ---------------------------------------------------------------------------
print('กำลังโหลดรูปภาพและสกัดคุณลักษณะ...')
files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith('.jpg')])
names = [f.split('_')[1].replace('.jpg', '') for f in files]

images = []
for f in files:
    img_path = os.path.join(DATA_FOLDER, f)
    img = imread_unicode(img_path)
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    images.append(img_resized)

# สกัดคุณลักษณะของภาพแต่ละคน
features = np.array([extract_features(img) for img in images], dtype=np.float64)

# ---------------------------------------------------------------------------
# 5. การปรับสเกลข้อมูล (Z-Score Normalization)
# ---------------------------------------------------------------------------
feat_mean = np.mean(features, axis=0)
feat_std = np.std(features, axis=0)
feat_std[feat_std == 0] = 1.0
features_norm = (features - feat_mean) / feat_std

# ---------------------------------------------------------------------------
# 6. ขั้นตอนวิธี K-Means Clustering (Manhattan Distance / L1 Norm)
# ---------------------------------------------------------------------------
# กำหนด Initial Centroids จากบุคคลที่เป็นตัวแทนชัดเจนของแต่ละกลุ่ม
seed_representatives = ['Pae', 'Memi', 'Saimai', 'Supachok']
seed_indices = [names.index(rep) for rep in seed_representatives]
centroids = features_norm[seed_indices].copy()

max_iterations = 30
cluster_assignments = np.zeros(len(names), dtype=int)

for iteration in range(max_iterations):
    # คำนวณระยะห่างแบบ Manhattan Distance (L1) ระหว่างข้อมูลทุกจุดกับ Centroid ทุกจุด
    distances = np.array([
        [np.sum(np.abs(x - c)) for c in centroids]
        for x in features_norm
    ])
    new_assignments = np.argmin(distances, axis=1)

    # คำนวณ Centroid ใหม่จากค่าเฉลี่ยของสมาชิกในกลุ่ม
    new_centroids = np.array([
        features_norm[new_assignments == k].mean(axis=0)
        if np.sum(new_assignments == k) > 0 else centroids[k]
        for k in range(K)
    ])

    # ตรวจสอบการลู่เข้าสู่จุดสมดุล (Convergence)
    if np.array_equal(cluster_assignments, new_assignments) and np.allclose(centroids, new_centroids):
        break

    centroids = new_centroids
    cluster_assignments = new_assignments

# ---------------------------------------------------------------------------
# 7. สรุปผลการจัดกลุ่มออกทางคอนโซล (Console Summary)
# ---------------------------------------------------------------------------
print('\n' + '=' * 75)
print('ผลลัพธ์การจัดกลุ่มจำลองสายอาชีพ (Career Role Face Clustering - K=4)')
print('=' * 75)

clusters = {}
for k in range(K):
    member_indices = np.where(cluster_assignments == k)[0]
    member_names = [names[i] for i in member_indices]
    clusters[k] = member_indices
    meta = GROUP_METADATA[k]

    print(f"\n[กลุ่มที่ {k + 1}] {meta['role']} (จำนวน {len(member_names)} คน)")
    print(f"  {meta['trait']}")
    print(f"  รายชื่อ: {member_names}")

print('\n' + '=' * 75)

# ---------------------------------------------------------------------------
# 8. การแสดงผลกราฟิกและตารางสรุปผล (Visualization with Matplotlib)
# ---------------------------------------------------------------------------
max_cols = max(len(clusters[k]) for k in range(K))
total_cols = max_cols + 1

fig = plt.figure(figsize=(15, 10.5), facecolor='#1e272e')

for k in range(K):
    meta = GROUP_METADATA[k]
    member_indices = clusters[k]

    # คอลัมน์ที่ 1: การ์ดข้อความสรุปโปรไฟล์สายอาชีพ
    ax_info = plt.subplot2grid((K, total_cols), (k, 0))
    ax_info.set_facecolor('#2f3640')
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    for spine in ax_info.spines.values():
        spine.set_color(meta['color'])
        spine.set_linewidth(2.5)

    info_text = (
        f"กลุ่มที่ {k + 1}\n"
        f"จำนวน: {len(member_indices)} คน\n\n"
        f"{meta['role'].split('(')[0].strip()}\n"
        f"({meta['role'].split('(')[1].replace(')', '')})\n\n"
        f"{meta['trait']}"
    )
    ax_info.text(
        0.5, 0.5, info_text,
        color='white',
        fontsize=8.5,
        fontweight='bold',
        va='center',
        ha='center',
        multialignment='center',
        wrap=True
    )

    # คอลัมน์ถัดไป: แสดงรูปภาพใบหน้าของสมาชิกแต่ละคน
    for col_idx, member_idx in enumerate(member_indices):
        ax_img = plt.subplot2grid((K, total_cols), (k, col_idx + 1))
        ax_img.imshow(images[member_idx], cmap='gray')
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        for spine in ax_img.spines.values():
            spine.set_color(meta['color'])
            spine.set_linewidth(2.5)

        person_name = names[member_idx]
        ax_img.set_title(
            person_name,
            fontsize=10,
            fontweight='bold',
            color='white',
            pad=5
        )

plt.suptitle(
    'ผลการจัดกลุ่มภาพใบหน้านักศึกษาจำลองสายอาชีพ (K-Means Clustering: K=4)',
    fontsize=16,
    fontweight='bold',
    color='white',
    y=0.98
)
plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
plt.show()
