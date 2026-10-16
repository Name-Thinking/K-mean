import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ตั้งค่าฟอนต์รองรับภาษาไทยในกราฟ
plt.rcParams['font.sans-serif'] = ['Tahoma', 'Leelawadee UI', 'Segoe UI', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ---- Settings ----
BASE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(BASE, 'หน้าใบรวม.py', 'ML_Data_บ่าย')
K = 4          # 4 กลุ่ม: ผู้ชาย, หญิงเปิดหน้าผาก, หญิงผมหน้าม้า, คนใส่แว่น
IMG_W = 140    # ปรับขนาดภาพให้เท่ากัน
IMG_H = 180

# ---- Load Images ----
files = sorted([f for f in os.listdir(FOLDER) if f.endswith('.jpg')])
names = [f.split('_')[1].replace('.jpg', '') for f in files]

def imread_unicode(path):
    """โหลดรูปภาพรองรับภาษาไทยด้วย numpy + cv2.imdecode"""
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

images = []
for f in files:
    img = imread_unicode(os.path.join(FOLDER, f))
    img = cv2.resize(img, (IMG_W, IMG_H))
    images.append(img)

print(f'Loaded {len(images)} face images')
print('Names:', names)

# ---- Feature Extraction (10 Features เพื่อแยก 4 กลุ่ม) ----
def extract_features(img):
    h, w = img.shape

    # 1. แว่นตา: ความมืดต่ำสุดแถบดวงตา (แถว 35-65%)
    eye = img[int(h*0.35):int(h*0.65), int(w*0.05):int(w*0.95)]
    row_means_eye = np.mean(eye, axis=1)
    f1 = float(np.min(row_means_eye))

    # 2. แว่นตา: คอนทราสต์กรอบแว่น (max - min ในแถบดวงตา)
    f2 = float(np.max(row_means_eye) - np.min(row_means_eye))

    # 3. ผมหน้าม้า vs เปิดหน้าผาก: ความสว่างหน้าผากส่วนบน (แถว 10-30%)
    #    เปิดหน้าผาก = ผิวสว่างมาก / ผมหน้าม้า = มืดจากเส้นผม
    forehead_top = img[int(h*0.10):int(h*0.30), int(w*0.20):int(w*0.80)]
    f3 = float(np.mean(forehead_top))

    # 4. ผมหน้าม้า: ความต่างหน้าผากบนเทียบหน้าผากล่าง (Hairline Contrast)
    forehead_mid = img[int(h*0.25):int(h*0.40), int(w*0.20):int(w*0.80)]
    f4 = float(np.mean(forehead_mid) - np.mean(forehead_top))

    # 5. โครงหน้า/เพศ: ความเข้มของคิ้ว (แถว 30-45%)
    #    ผู้ชายคิ้วเข้มหนากว่าผู้หญิง
    eyebrow_zone = img[int(h*0.30):int(h*0.45), :]
    f5 = float(np.min(np.mean(eyebrow_zone, axis=1)))

    # 6. เพศ/โครงหน้า: ความสว่างและเงาช่วงกรามล่าง (แถว 70-95%)
    jaw_zone = img[int(h*0.70):int(h*0.95), :]
    f6 = float(np.mean(jaw_zone))

    # 7. โทนสีผิวโดยรวม (Overall Brightness)
    f7 = float(np.mean(img))

    # 8. ตำแหน่งขอบผมด้านบน (Hairline Row Position)
    row_means_all = np.mean(img, axis=1)
    thresh = np.mean(img) - np.std(img)
    dark_rows = np.where(row_means_all < thresh)[0]
    f8 = float(dark_rows[0]) if len(dark_rows) > 0 else 0.0

    # 9. ความสมมาตรซ้าย-ขวา (Left-Right Symmetry)
    f9 = float(abs(np.mean(img[:, :w//2]) - np.mean(img[:, w//2:])))

    # 10. ความแปรปรวนบริเวณดวงตา (Eye Texture Variance)
    f10 = float(np.var(eye))

    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]

X = np.array([extract_features(img) for img in images])
print(f'\nFeature matrix shape: {X.shape}  (18 images x 10 features)')

# ---- Normalize (Mean=0, Std=1) ----
X_mean = np.mean(X, axis=0)
X_std  = np.std(X, axis=0)
X_std[X_std == 0] = 1.0
X_norm = (X - X_mean) / X_std

# ---- ให้น้ำหนัก Feature ตามเกณฑ์ 4 กลุ่ม ----
# Feature 0, 1 (แว่นตา) ให้น้ำหนัก 3.5x เพื่อรวมคนใส่แว่น
X_norm[:, 0] *= 3.5
X_norm[:, 1] *= 3.5

# Feature 2, 3 (ผมหน้าม้า/หน้าผาก) ให้น้ำหนัก 2.5x เพื่อแยกผมหน้าม้าออกจากเปิดหน้าผาก
X_norm[:, 2] *= 2.5
X_norm[:, 3] *= 2.5

# Feature 4, 5 (คิ้ว/กราม) ให้น้ำหนัก 1.5x เพื่อแยกเพศชาย
X_norm[:, 4] *= 1.5
X_norm[:, 5] *= 1.5

# ---- กำหนด Seed Centroid เริ่มต้นตามตัวแทน 4 กลุ่ม (Informed Seeding) ----
# 1. ผู้ชาย (Men): Pae
# 2. ผู้หญิงเปิดหน้าผาก (Women - Open Forehead): Memi
# 3. ผู้หญิงผมหน้าม้า (Women - Bangs): Saimai
# 4. คนใส่แว่นตา (Glasses): Manow
archetypes = ['Pae', 'Memi', 'Saimai', 'Manow']
init_indices = [names.index(a) if a in names else i for i, a in enumerate(archetypes)]
C = X_norm[init_indices].copy()

# ---- K-Means (Manhattan Distance, K=4) ----
print('\nRunning K-Means (Manhattan distance, K=4)...')
for iteration in range(300):
    # คำนวณ Manhattan Distance (L1 norm)
    D = np.zeros((len(X_norm), K))
    for k in range(K):
        D[:, k] = np.sum(np.abs(X_norm - C[k]), axis=1)

    idx = np.argmin(D, axis=1)

    # อัปเดต Centroid ใหม่
    Cold = C.copy()
    for k in range(K):
        if np.sum(idx == k) > 0:
            C[k] = np.mean(X_norm[idx == k], axis=0)

    # หยุดเมื่อ Centroid ไม่เปลี่ยนตำแหน่ง
    if np.sum(np.abs(Cold - C)) == 0:
        print(f'Converged at iteration {iteration + 1}')
        break

# ข้อมูลชื่อกลุ่มและสี
GROUP_INFO = [
    {'title': 'Group 1: ผู้ชาย (Men)',              'color': '#3498db'}, # ฟ้า
    {'title': 'Group 2: หญิง-เปิดหน้าผาก (Women)',   'color': '#e84393'}, # ชมพู
    {'title': 'Group 3: หญิง-ผมหน้าม้า (Bangs)',     'color': '#e67e22'}, # ส้ม
    {'title': 'Group 4: บุคคลใส่แว่น (Glasses)',    'color': '#2ecc71'}  # เขียว
]

# ---- แสดงผลใน Terminal ----
print('\n' + '='*55)
print('        ผลการจัดกลุ่มใบหน้าด้วย K-Means (K=4)')
print('='*55)
for k in range(K):
    members = [names[i] for i in range(len(names)) if idx[i] == k]
    print(f"\n  {GROUP_INFO[k]['title']}  ({len(members)} คน)")
    print('  ' + '─'*35)
    for m in members:
        print(f'      • {m}')

print('\n' + '='*55)
print('สรุปรายบุคคล:')
print(f"  {'ชื่อ':<15} {'กลุ่ม'}")
print('  ' + '─'*35)
for i, name in enumerate(names):
    grp_title = GROUP_INFO[idx[i]]['title'].split(':')[1].strip()
    print(f"  {name:<15} Group {idx[i]+1} ({grp_title})")
print('='*55)

# ---- แสดงกราฟตารางภาพใบหน้าแยกแต่ละกลุ่ม ----
max_per_row = max(int(np.sum(idx == k)) for k in range(K))
max_per_row = max(max_per_row, 1)

fig, axes = plt.subplots(K, max_per_row,
                         figsize=(max_per_row * 2.1 + 3.0, K * 3.0))
fig.patch.set_facecolor('#1a1a2e')

# เว้นขอบซ้ายสำหรับป้ายชื่อกลุ่มภาษาไทย
plt.subplots_adjust(left=0.24, right=0.98, top=0.92, bottom=0.03,
                    hspace=0.45, wspace=0.15)

if K == 1:
    axes = np.array([axes])
elif max_per_row == 1:
    axes = axes.reshape(K, 1)

for k in range(K):
    members_idx = [i for i in range(len(names)) if idx[i] == k]
    color = GROUP_INFO[k]['color']
    title_text = GROUP_INFO[k]['title']

    for j in range(max_per_row):
        ax = axes[k][j]
        ax.set_facecolor('#1a1a2e')

        if j < len(members_idx):
            ax.imshow(images[members_idx[j]], cmap='gray')
            ax.set_title(names[members_idx[j]], fontsize=9,
                         color='white', fontweight='bold', pad=3)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3.5)
                spine.set_visible(True)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # ซ่อนช่องว่างที่ไม่มีรูป
            ax.axis('off')

    # ใส่ป้ายชื่อกลุ่มด้านซ้ายของแต่ละแถว
    axes[k][0].text(-0.32, 0.5, title_text,
                    transform=axes[k][0].transAxes,
                    fontsize=12, color=color, fontweight='bold',
                    va='center', ha='right')

plt.suptitle('Face Clustering by K-Means (K=4 Groups)',
             fontsize=15, fontweight='bold', color='white', y=0.97)

print('\nกำลังแสดงหน้าต่างกราฟ...')
plt.show()
