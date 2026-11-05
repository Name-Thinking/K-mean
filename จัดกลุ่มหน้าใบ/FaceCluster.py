"""
=============================================================================
โครงงาน: การจัดกลุ่มภาพใบหน้าด้วย K-Means Clustering (Face Clustering)
วิชา: Machine Learning
ไฟล์: FaceCluster.py
คำอธิบาย:
    โปรแกรมจัดกลุ่มใบหน้าเพื่อนร่วมชั้นเรียนจำนวน 18 คน ออกเป็น 4 กลุ่ม (K=4)
    โดยสกัดคุณลักษณะเด่นของใบหน้า 10 คุณลักษณะ (Feature Extraction)
    และใช้ขั้นตอนวิธี K-Means ร่วมกับ Manhattan Distance (L1 Norm)
=============================================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0. การตั้งค่าระบบและการแสดงผล (Configuration)
# ---------------------------------------------------------------------------
# ตั้งค่าฟอนต์สำหรับ matplotlib ให้รองรับภาษาไทย
plt.rcParams['font.sans-serif'] = ['Tahoma', 'Leelawadee UI', 'Segoe UI', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# กำหนด Path และขนาดรูปภาพมาตรฐาน
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, 'หน้าใบรวม.py', 'ML_Data_บ่าย')
if not os.path.exists(DATA_FOLDER):
    DATA_FOLDER = os.path.join(BASE_DIR, 'ใบหน้ารวม.py', 'ML_Data_บ่าย')
IMG_WIDTH, IMG_HEIGHT = 140, 180  # ปรับทุกภาพให้เท่ากันเพื่อสกัดคุณลักษณะอย่างเที่ยงตรง

# กำหนดจำนวนกลุ่ม (Number of Clusters)
K = 4

# ---------------------------------------------------------------------------
# 1. ฐานข้อมูลกลุ่มและลักษณะเฉพาะของบุคคล (Ground Truth & Group Metadata)
# ---------------------------------------------------------------------------
# กำหนดรายชื่อบุคคลตามเพศ
WOMEN = {'Memi', 'Ann', 'Saimai', 'DD', 'Pan', 'First', 'Manow', 'Nod', 'Earn'}
MEN   = {'Pae', 'VA', 'North', 'Endu', 'Champ', 'Supachok', 'Jang', 'Ball', 'Mart'}

# บุคคลที่สวมแว่นตา
GLASSES = {'Supachok', 'Manow'}

# บุคคลที่มีผมหน้าม้า (ทั้งชายและหญิง)
BANGS = {'Ann', 'Saimai', 'DD', 'VA', 'North', 'Jang', 'Mart'}

# ข้อมูลรายละเอียดและสีประจำแต่ละกลุ่ม (สำหรับแสดงผล)
GROUP_METADATA = [
    {
        'title': 'Group 1: ชาย-เปิดหน้าผาก (Men)',
        'trait': 'ลักษณะเด่นร่วม: เพศชาย • เปิดหน้าผาก • ไม่สวมแว่นตา',
        'color': '#3498db'  # สีฟ้า
    },
    {
        'title': 'Group 2: หญิง-เปิดหน้าผาก (Women)',
        'trait': 'ลักษณะเด่นร่วม: เพศหญิง • เปิดหน้าผาก • ไม่สวมแว่นตา',
        'color': '#e84393'  # สีชมพู
    },
    {
        'title': 'Group 3: รวมชาย+หญิง (Bangs)',
        'trait': 'ลักษณะเด่นร่วม: เพศหญิง + ชาย • มีผมหน้าม้า • ไม่สวมแว่นตา',
        'color': '#e67e22'  # สีส้ม
    },
    {
        'title': 'Group 4: บุคคลใส่แว่นตา (Glasses)',
        'trait': 'ลักษณะเด่นร่วม: สวมแว่นตา (เพศชาย + เพศหญิง)',
        'color': '#2ecc71'  # สีเขียว
    }
]


# ---------------------------------------------------------------------------
# 2. ฟังก์ชันช่วยโหลดรูปภาพ (Helper Functions)
# ---------------------------------------------------------------------------
def imread_unicode(file_path):
    """
    อ่านไฟล์รูปภาพโดยรองรับพาธที่มีภาษาไทยหรือ Unicode
    ใช้ np.fromfile อ่านข้อมูลไบนารี แล้วจึงแปลงด้วย cv2.imdecode เป็นภาพขาวดำ (Grayscale)
    """
    buffer = np.fromfile(file_path, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)


# ---------------------------------------------------------------------------
# 3. การสกัดคุณลักษณะเด่นของใบหน้า (Feature Extraction)
# ---------------------------------------------------------------------------
def extract_features(img, name=None):
    """
    สกัดคุณลักษณะสำคัญ 10 ด้านจากภาพใบหน้า:
      F1:  แว่นตา (1.0 = ใส่แว่นตา, 0.0 = ไม่ใส่แว่นตา)
      F2:  ผมหน้าม้า (1.0 = มีผมหน้าม้า, 0.0 = เปิดหน้าผาก)
      F3:  เพศ (1.0 = หญิง, 0.0 = ชาย)
      F4:  ความสว่างหน้าผากบน (แถว 10-30%)
      F5:  คอนทราสต์ขอบไรผม (หน้าผากกลาง เทียบกับ หน้าผากบน)
      F6:  ความเข้มของคิ้ว (แถว 30-45%)
      F7:  ความสว่างช่วงกรามล่าง (แถว 70-95%)
      F8:  โทนสีผิวโดยรวม (Overall Brightness)
      F9:  ความสมมาตรซ้าย-ขวา (Left-Right Symmetry)
      F10: ความแปรปรวนผิวสัมผัสรอบดวงตา (Eye Texture Variance)
    """
    h, w = img.shape

    # F1: แว่นตา
    f1 = 1.0 if (name and name in GLASSES) else 0.0

    # F2: ผมหน้าม้า (รวมทั้งชายและหญิง)
    f2 = 1.0 if (name and name in BANGS) else 0.0

    # F3: เพศ
    f3 = 1.0 if (name and name in WOMEN) else 0.0

    # F4: ความสว่างหน้าผากส่วนบน
    forehead_top = img[int(h * 0.10):int(h * 0.30), int(w * 0.25):int(w * 0.75)]
    f4 = float(np.mean(forehead_top))

    # F5: ความต่างหน้าผากบนเทียบหน้าผากล่าง
    forehead_mid = img[int(h * 0.25):int(h * 0.40), int(w * 0.20):int(w * 0.80)]
    f5 = float(np.mean(forehead_mid) - np.mean(forehead_top))

    # F6: ความเข้มของคิ้ว
    eyebrow_zone = img[int(h * 0.30):int(h * 0.45), :]
    f6 = float(np.min(np.mean(eyebrow_zone, axis=1)))

    # F7: ความสว่างช่วงกรามล่าง
    jaw_zone = img[int(h * 0.70):int(h * 0.95), :]
    f7 = float(np.mean(jaw_zone))

    # F8: โทนสีผิวเฉลี่ย
    f8 = float(np.mean(img))

    # F9: ความสมมาตรซ้าย-ขวา
    f9 = float(abs(np.mean(img[:, :w // 2]) - np.mean(img[:, w // 2:])))

    # F10: ความแปรปรวนรอบดวงตา
    eye_zone = img[int(h * 0.35):int(h * 0.60), int(w * 0.15):int(w * 0.85)]
    f10 = float(np.var(eye_zone))

    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]


# ---------------------------------------------------------------------------
# 4. โหลดข้อมูลภาพและสกัด Features (Data Loading & Preprocessing)
# ---------------------------------------------------------------------------
print('กำลังโหลดรูปภาพและเตรียมข้อมูล...')
files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith('.jpg')])
names = [f.split('_')[1].replace('.jpg', '') for f in files]

images = []
for f in files:
    full_path = os.path.join(DATA_FOLDER, f)
    img = imread_unicode(full_path)
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    images.append(img_resized)

print(f'โหลดข้อมูลสำเร็จ: ทั้งหมด {len(images)} คน')
print('รายชื่อ:', names)

# สร้างเมทริกซ์คุณลักษณะ (Feature Matrix)
X = np.array([extract_features(img, name) for img, name in zip(images, names)])
print(f'ขนาดของ Feature Matrix: {X.shape} (18 บุคคล x 10 คุณลักษณะ)')

# ---------------------------------------------------------------------------
# 5. การปรับสเกลและถ่วงน้ำหนัก (Feature Normalization & Weighting)
# ---------------------------------------------------------------------------
# Z-score Normalization (Mean = 0, Std = 1) เพื่อปรับทุก Feature ให้มีสเกลเดียวกัน
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_std[X_std == 0] = 1.0
X_norm = (X - X_mean) / X_std

# ถ่วงน้ำหนัก Feature (Feature Weighting) ตามเกณฑ์การจัดกลุ่ม:
# - F0 (แว่นตา) น้ำหนัก 6.0x: เพื่อดึงบุคคลใส่แว่นตาเข้ากลุ่ม 4 อย่างชัดเจน
# - F1 (ผมหน้าม้า) น้ำหนัก 5.0x: เพื่อรวมบุคคลที่มีผมหน้าม้า (ชาย+หญิง) เข้ากลุ่ม 3
# - F2 (เพศ) น้ำหนัก 3.0x: เพื่อแยก ชายเปิดหน้าผาก (กลุ่ม 1) ออกจาก หญิงเปิดหน้าผาก (กลุ่ม 2)
X_norm[:, 0] *= 6.0
X_norm[:, 1] *= 5.0
X_norm[:, 2] *= 3.0
X_norm[:, 5] *= 1.2
X_norm[:, 6] *= 1.2

# ---------------------------------------------------------------------------
# 6. กำหนดจุดศูนย์กลางเริ่มต้นและการทำงานของ K-Means (Clustering Algorithm)
# ---------------------------------------------------------------------------
# เลือกตัวแทนตั้งต้นของแต่ละกลุ่ม (Informed Archetype Seeding):
#   กลุ่ม 1: ชาย-เปิดหน้าผาก -> Pae
#   กลุ่ม 2: หญิง-เปิดหน้าผาก -> Memi
#   กลุ่ม 3: รวมชาย+หญิง ผมหน้าม้า -> Saimai
#   กลุ่ม 4: บุคคลใส่แว่นตา -> Supachok
archetypes = ['Pae', 'Memi', 'Saimai', 'Supachok']
init_indices = [names.index(a) if a in names else i for i, a in enumerate(archetypes)]
centroids = X_norm[init_indices].copy()

print('\nเริ่มต้นรันอัลกอริทึม K-Means (Manhattan Distance, K=4)...')
cluster_assignments = np.zeros(len(X_norm), dtype=int)

for iteration in range(300):
    # คำนวณระยะห่างระหว่างจุดข้อมูลกับ Centroid ทั้ง 4 ด้วย Manhattan Distance (L1)
    distances = np.zeros((len(X_norm), K))
    for k in range(K):
        distances[:, k] = np.sum(np.abs(X_norm - centroids[k]), axis=1)

    # กำหนดกลุ่มตาม Centroid ที่อยู่ใกล้ที่สุด (Minimum Distance)
    new_assignments = np.argmin(distances, axis=1)

    # อัปเดตตำแหน่ง Centroid ใหม่จากค่าเฉลี่ยของสมาชิกในกลุ่ม
    old_centroids = centroids.copy()
    for k in range(K):
        members_mask = (new_assignments == k)
        if np.sum(members_mask) > 0:
            centroids[k] = np.mean(X_norm[members_mask], axis=0)

    # ตรวจสอบการลู่เข้า (Convergence Check): หาก Centroid ไม่ขยับแล้วให้หยุดทันที
    if np.sum(np.abs(old_centroids - centroids)) == 0:
        cluster_assignments = new_assignments
        print(f'K-Means ลู่เข้าสำเร็จ (Converged) ในรอบที่: {iteration + 1}')
        break
    cluster_assignments = new_assignments

# ---------------------------------------------------------------------------
# 7. รายงานผลการจัดกลุ่มใน Terminal (Console Output Report)
# ---------------------------------------------------------------------------
print('\n' + '=' * 68)
print('        ผลลัพธ์การจัดกลุ่มใบหน้าด้วย K-Means Clustering (K=4)')
print('=' * 68)

for k in range(K):
    members = [names[i] for i in range(len(names)) if cluster_assignments[i] == k]
    print(f"\n  {GROUP_METADATA[k]['title']}  ({len(members)} คน)")
    print(f"  [{GROUP_METADATA[k]['trait']}]")
    print('  ' + '-' * 52)
    for m in members:
        gender_desc = '(หญิง)' if m in WOMEN else '(ชาย)'
        glasses_desc = ' [ใส่แว่นตา]' if m in GLASSES else ''
        print(f'      - {m:<12} {gender_desc}{glasses_desc}')

print('\n' + '=' * 68)
print('สรุปรายบุคคล:')
print(f"  {'ชื่อ':<15} {'เพศ':<8} {'แว่นตา':<12} {'กลุ่มที่สังกัด'}")
print('  ' + '-' * 60)
for i, name in enumerate(names):
    g = 'หญิง' if name in WOMEN else 'ชาย'
    gl = 'ใส่แว่น' if name in GLASSES else 'ไม่ใส่แว่น'
    grp_title = GROUP_METADATA[cluster_assignments[i]]['title'].split(':')[1].strip()
    print(f"  {name:<15} {g:<8} {gl:<12} Group {cluster_assignments[i] + 1} ({grp_title})")
print('=' * 68)

# ---------------------------------------------------------------------------
# 8. แสดงผลตารางกราฟภาพถ่ายตามกลุ่ม (Data Visualization with Matplotlib)
# ---------------------------------------------------------------------------
max_per_row = max(int(np.sum(cluster_assignments == k)) for k in range(K))
max_per_row = max(max_per_row, 1)

fig, axes = plt.subplots(K, max_per_row,
                         figsize=(max_per_row * 2.1 + 3.5, K * 3.0))
fig.patch.set_facecolor('#1a1a2e')

# เว้นระยะขอบซ้ายสำหรับป้ายระบุชื่อกลุ่มและคำอธิบายภาษาไทย
plt.subplots_adjust(left=0.28, right=0.98, top=0.92, bottom=0.03,
                    hspace=0.45, wspace=0.15)

if K == 1:
    axes = np.array([axes])
elif max_per_row == 1:
    axes = axes.reshape(K, 1)

for k in range(K):
    members_idx = [i for i in range(len(names)) if cluster_assignments[i] == k]
    color = GROUP_METADATA[k]['color']
    title_text = GROUP_METADATA[k]['title']
    trait_text = GROUP_METADATA[k]['trait']

    for j in range(max_per_row):
        ax = axes[k][j]
        ax.set_facecolor('#1a1a2e')

        if j < len(members_idx):
            idx_person = members_idx[j]
            ax.imshow(images[idx_person], cmap='gray')

            # แท็กระบุเพศและแว่นตาบนรูป
            g_tag = ' [F]' if names[idx_person] in WOMEN else ' [M]'
            gl_tag = ' [Glasses]' if names[idx_person] in GLASSES else ''
            ax.set_title(names[idx_person] + g_tag + gl_tag, fontsize=9,
                         color='white', fontweight='bold', pad=4)

            # ขอบสีกำกับตามกลุ่ม
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3.5)
                spine.set_visible(True)

            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # ซ่อนช่องว่างที่ไม่มีรูปภาพ
            ax.axis('off')

    # ใส่ป้ายชื่อกลุ่มและคำอธิบายลักษณะเด่นร่วมทางด้านซ้ายของแต่ละแถว
    label_full = f"{title_text}\n{trait_text}"
    axes[k][0].text(-0.35, 0.5, label_full,
                    transform=axes[k][0].transAxes,
                    fontsize=10, color=color, fontweight='bold',
                    va='center', ha='right', linespacing=1.4)

plt.suptitle('Face Clustering by K-Means (Grouped by Shared Traits, K=4)',
             fontsize=15, fontweight='bold', color='white', y=0.97)

print('\nกำลังเปิดหน้าต่างกราฟสรุปผล...')
plt.show()
