# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
import os

# ---- Settings ----
BASE   = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(BASE, 'หน้าใบรวม.py', 'ML_Data_บ่าย')
if not os.path.exists(FOLDER):
    FOLDER = os.path.join(BASE, 'ใบหน้ารวม.py', 'ML_Data_บ่าย')
IMG_W, IMG_H = 140, 180          # resize all faces to the same size

# กำหนดกลุ่มเพศ, ผมหน้าม้า และบุคคลใส่แว่นตา
WOMEN   = {'Memi', 'Ann', 'Saimai', 'DD', 'Pan', 'First', 'Manow', 'Nod', 'Earn'}
MEN     = {'Pae', 'VA', 'North', 'Endu', 'Champ', 'Supachok', 'Jang', 'Ball', 'Mart'}
GLASSES = {'Supachok', 'Manow'}
BANGS   = {'Ann', 'Saimai', 'DD', 'VA', 'North', 'Jang', 'Mart'}

PERSON_INFO = {
    # Group 1: ชาย-เปิดหน้าผาก (4 คน)
    'Pae':      {'gender': 'ชาย',  'glasses': False, 'group_th': 'กลุ่ม 1: ชาย-เปิดหน้าผาก',       'group_en': 'G1: Men-Forehead'},
    'Endu':     {'gender': 'ชาย',  'glasses': False, 'group_th': 'กลุ่ม 1: ชาย-เปิดหน้าผาก',       'group_en': 'G1: Men-Forehead'},
    'Champ':    {'gender': 'ชาย',  'glasses': False, 'group_th': 'กลุ่ม 1: ชาย-เปิดหน้าผาก',       'group_en': 'G1: Men-Forehead'},
    'Ball':     {'gender': 'ชาย',  'glasses': False, 'group_th': 'กลุ่ม 1: ชาย-เปิดหน้าผาก',       'group_en': 'G1: Men-Forehead'},

    # Group 2: หญิง-เปิดหน้าผาก (5 คน)
    'Memi':     {'gender': 'หญิง', 'glasses': False, 'group_th': 'กลุ่ม 2: หญิง-เปิดหน้าผาก',      'group_en': 'G2: Women-Forehead'},
    'Pan':      {'gender': 'หญิง', 'glasses': False, 'group_th': 'กลุ่ม 2: หญิง-เปิดหน้าผาก',      'group_en': 'G2: Women-Forehead'},
    'First':    {'gender': 'หญิง', 'glasses': False, 'group_th': 'กลุ่ม 2: หญิง-เปิดหน้าผาก',      'group_en': 'G2: Women-Forehead'},
    'Nod':      {'gender': 'หญิง', 'glasses': False, 'group_th': 'กลุ่ม 2: หญิง-เปิดหน้าผาก',      'group_en': 'G2: Women-Forehead'},
    'Earn':     {'gender': 'หญิง', 'glasses': False, 'group_th': 'กลุ่ม 2: หญิง-เปิดหน้าผาก',      'group_en': 'G2: Women-Forehead'},

    # Group 3: รวมชาย+หญิง ผมหน้าม้า (7 คน)
    'VA':       {'gender': 'ชาย',  'glasses': False, 'group_th': 'กลุ่ม 3: รวมชาย+หญิง ผมหน้าม้า', 'group_en': 'G3: Bangs'},
    'North':    {'gender': 'ชาย',  'glasses': False, 'group_th': 'กลุ่ม 3: รวมชาย+หญิง ผมหน้าม้า', 'group_en': 'G3: Bangs'},
    'Jang':     {'gender': 'ชาย',  'glasses': False, 'group_th': 'กลุ่ม 3: รวมชาย+หญิง ผมหน้าม้า', 'group_en': 'G3: Bangs'},
    'Mart':     {'gender': 'ชาย',  'glasses': False, 'group_th': 'กลุ่ม 3: รวมชาย+หญิง ผมหน้าม้า', 'group_en': 'G3: Bangs'},
    'Ann':      {'gender': 'หญิง', 'glasses': False, 'group_th': 'กลุ่ม 3: รวมชาย+หญิง ผมหน้าม้า', 'group_en': 'G3: Bangs'},
    'Saimai':   {'gender': 'หญิง', 'glasses': False, 'group_th': 'กลุ่ม 3: รวมชาย+หญิง ผมหน้าม้า', 'group_en': 'G3: Bangs'},
    'DD':       {'gender': 'หญิง', 'glasses': False, 'group_th': 'กลุ่ม 3: รวมชาย+หญิง ผมหน้าม้า', 'group_en': 'G3: Bangs'},

    # Group 4: บุคคลใส่แว่นตา (2 คน)
    'Supachok': {'gender': 'ชาย',  'glasses': True,  'group_th': 'กลุ่ม 4: บุคคลใส่แว่นตา',         'group_en': 'G4: Glasses'},
    'Manow':    {'gender': 'หญิง', 'glasses': True,  'group_th': 'กลุ่ม 4: บุคคลใส่แว่นตา',         'group_en': 'G4: Glasses'},
}

# ---- Helper: read image with Thai/Unicode path ----
def imread_unicode(path):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

# ---- Extract features ----
def extract_features(img):
    img = cv2.resize(img, (IMG_W, IMG_H))
    h, w = img.shape

    eye_band = img[int(h*0.40):int(h*0.70), :]

    # 1. Glasses region darkness
    f1 = float(np.mean(eye_band))
    # 2. Forehead brightness (top 30%)
    f2 = float(np.mean(img[:int(h*0.30), :]))
    # 3. Face aspect ratio
    f3 = float(w / h)
    # 4. Eyebrow darkness - darkest row in top 50%
    f4 = float(np.min(np.mean(img[:int(h*0.50), :], axis=1)))
    # 5. Skin tone - overall brightness
    f5 = float(np.mean(img))
    # 6. Hairline position - first dark row from top
    threshold = np.mean(img) - np.std(img)
    dark_rows = np.where(np.mean(img, axis=1) < threshold)[0]
    f6 = float(dark_rows[0]) if len(dark_rows) > 0 else 0.0
    # 7. Left-right symmetry
    f7 = float(abs(np.mean(img[:, :w//2]) - np.mean(img[:, w//2:])))
    # 8. Nose bridge width - min col mean in center
    nose = img[int(h*0.40):int(h*0.65), int(w*0.35):int(w*0.65)]
    f8 = float(np.min(np.mean(nose, axis=0)))
    # 9. Face edge sharpness
    border = np.concatenate([img[:, :5].flatten(), img[:, -5:].flatten()])
    f9 = float(np.std(border))
    # 10. Eye openness - variance in eye band
    f10 = float(np.var(eye_band))

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10], dtype=float)

# ================================================================
# STEP 1: Load training images & extract features
# ================================================================
print('Loading training data ...')
files = sorted([f for f in os.listdir(FOLDER) if f.endswith('.jpg')])
names = [f.split('_')[1].replace('.jpg', '') for f in files]

train_feats = np.array([
    extract_features(imread_unicode(os.path.join(FOLDER, f)))
    for f in files
])

# Normalize (mean=0, std=1)
X_mean = np.mean(train_feats, axis=0)
X_std  = np.std(train_feats, axis=0)
X_std[X_std == 0] = 1
train_norm = (train_feats - X_mean) / X_std

print(f'Ready! {len(names)} people loaded.')
print("  Group 1: ชาย-เปิดหน้าผาก (4 คน): ['Pae', 'Endu', 'Champ', 'Ball']")
print("  Group 2: หญิง-เปิดหน้าผาก (5 คน): ['Memi', 'Pan', 'First', 'Nod', 'Earn']")
print("  Group 3: รวมชาย+หญิง ผมหน้าม้า (7 คน): ['VA', 'North', 'Jang', 'Mart', 'Ann', 'Saimai', 'DD']")
print("  Group 4: บุคคลใส่แว่นตา (2 คน): ['Supachok [M]', 'Manow [F]']")
print('Press  Q  to quit.')

# ================================================================
# STEP 2: Webcam — scan face → identify person
# ================================================================

# ---- Auto-detect camera index ----
cap = None
for cam_idx in range(4):          # ลอง index 0, 1, 2, 3
    print(f'Trying camera index {cam_idx} ...')
    test = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if test.isOpened():
        ret_test, _ = test.read()
        if ret_test:
            cap = test
            print(f'Camera {cam_idx} opened OK!')
            break
        test.release()
    else:
        test.release()

if cap is None:
    print('ERROR: ไม่พบกล้อง! ลองทำสิ่งต่อไปนี้:')
    print('  1. ปิด Teams / Zoom / Google Meet ที่อาจใช้กล้องอยู่')
    print('  2. ถอดแล้วเสียบ USB กล้องใหม่')
    print('  3. ตรวจสอบ Device Manager ว่ากล้องมีอยู่')
    exit(1)

# Same crop region as FaceTrain.py
X1, Y1, X2, Y2 = 250, 120, 390, 300
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print('กล้องหยุดส่งภาพ — ออกจากโปรแกรม')
        break

    # Crop & convert face region to grayscale
    face_gray = cv2.cvtColor(frame[Y1:Y2, X1:X2], cv2.COLOR_BGR2GRAY)

    # ---- 1-Nearest Neighbor (get Top 3) ----
    feat      = extract_features(face_gray)
    feat_norm = (feat - X_mean) / X_std
    distances = np.sum(np.abs(train_norm - feat_norm), axis=1)  # Manhattan

    top3_idx  = np.argsort(distances)[:3]   # index ของ 3 คนใกล้สุด
    best_name = names[top3_idx[0]]
    best_dist = distances[top3_idx[0]]
    best_info = PERSON_INFO.get(best_name, {'gender': '?', 'glasses': False, 'group_th': '?', 'group_en': '?'})
    is_female = best_name in WOMEN
    has_glasses = best_name in GLASSES

    # ---- Color by group / gender & confidence ----
    if has_glasses:
        color = (0, 255, 0)      # เขียว = บุคคลใส่แว่นตา
    elif best_info['group_en'] == 'G3: Bangs':
        color = (0, 140, 255)    # ส้ม = กลุ่มผมหน้าม้า
    elif is_female:
        color = (203, 108, 255)  # ชมพู = หญิงเปิดหน้าผาก
    else:
        color = (255, 180, 0)    # ฟ้า = ชายเปิดหน้าผาก

    # ---- Draw face box + name + gender + group ----
    cv2.rectangle(frame, (X1, Y1), (X2, Y2), color, 2)
    gender_tag = "[F]" if is_female else "[M]"
    glasses_tag = " (Glasses)" if has_glasses else ""
    display_title = f"{best_name} {gender_tag}{glasses_tag}"
    cv2.putText(frame, display_title,
                (X1, Y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(frame, best_info['group_en'],
                (X1, Y1 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f'dist={best_dist:.1f}',
                (X1, Y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)

    # ---- Confidence bar (ยิ่ง dist น้อย ยิ่งยาว) ----
    bar_max   = 200
    bar_len   = max(0, int(bar_max - best_dist * 3))
    cv2.rectangle(frame, (X1, Y2 + 8), (X1 + bar_max, Y2 + 22), (50,50,50), -1)
    cv2.rectangle(frame, (X1, Y2 + 8), (X1 + bar_len,  Y2 + 22), color, -1)
    cv2.putText(frame, f"confidence ({best_info['gender']})", (X1, Y2 + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    # ---- Top 3 ranking (มุมซ้ายบน) ----
    cv2.putText(frame, 'Top 3:', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    for rank, i in enumerate(top3_idx):
        cand_name = names[i]
        cand_gender = '[F]' if cand_name in WOMEN else '[M]'
        cand_glasses = ' [Glasses]' if cand_name in GLASSES else ''
        rank_color = (0, 255, 0) if rank == 0 else (180, 180, 180)
        cv2.putText(frame, f'{rank+1}. {cand_name} {cand_gender}{cand_glasses} ({distances[i]:.1f})',
                    (10, 50 + rank * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, rank_color, 1)

    # ---- Print to console every ~30 frames ----
    frame_count += 1
    if frame_count % 30 == 0:
        c2 = names[top3_idx[1]]
        c3 = names[top3_idx[2]]
        g_desc = f" | {best_info['gender']}"
        gl_desc = ' (ใส่แว่น)' if has_glasses else ''
        print(f"[Recognized] {best_name:<10}{gl_desc}{g_desc} | {best_info['group_th']} | "
              f"dist={best_dist:.2f} | 2nd: {c2} ({distances[top3_idx[1]]:.1f}) | 3rd: {c3} ({distances[top3_idx[2]]:.1f})")

    # ---- Show windows ----
    face_bgr_show = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)
    cv2.imshow('Face Recognition  (Q=quit)', frame)
    cv2.imshow('Face', face_bgr_show)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
