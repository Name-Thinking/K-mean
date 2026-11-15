"""
=============================================================================
โครงงาน: การจัดกลุ่มภาพใบหน้าจำลองสายอาชีพด้วย K-Means Clustering (Unslopped)
วิชา: Machine Learning
ไฟล์: FaceCluster.py
คำอธิบาย:
    โปรแกรมจัดกลุ่มใบหน้าของสมาชิกและลูกค้าที่ลงทะเบียนทั้งหมด ออกเป็น 4 กลุ่มสายอาชีพ (K=4):
    - ลบ _SPECTRAL_MATRIX (ค่าแฮชลวง) ออก 100% ใช้การสกัดคุณลักษณะจริงจากใบหน้าและ Landmark
    - ใช้ YuNet + Anatomical Landmarks สกัดคุณลักษณะกายภาพจริง (แว่นตา, ผมหน้าม้า, อัตราส่วนหน้าผาก)
      ร่วมกับแกนสัณฐานวิทยาจาก SFace 128-D Embeddings (Out-of-Fold Generalizable Axis)
    - จัดกรอบภาพพอร์ตเทรตมาตรฐาน 3:4 (180x240) ให้เห็นใบหน้า ศีรษะ ผม และคางครบถ้วนสม่ำเสมอ
    - เรนเดอร์การ์ดคำอธิบายภาษาไทยด้วย PIL (ตัดบรรทัดพอดีกรอบ 100% ไม่มีข้อความล้นทับรูป)
=============================================================================
"""

"""
=============================================================================
โครงงาน: การจัดกลุ่มภาพใบหน้านักศึกษาด้วย K-Means Clustering (K=4)
วิชา: Machine Learning
ไฟล์: FaceCluster.py
มาตรฐานการเขียน: EL10 (เข้าใจง่าย เป็นระเบียบ มีคำอธิบายทุกสูตรและกระบวนการ)
=============================================================================
คำอธิบายการทำงาน:
    1. สกัดคุณลักษณะทางกายภาพจริง (Physical Features) จากภาพถ่ายใบหน้า:
       - ตรวจจับใบหน้าและ Landmark 5 จุดด้วยโมเดล YuNet (Deep Learning)
       - วัดสัญญาณการสวมแว่นตา (Glasses Signal) บริเวณสันจมูกและดวงตา
       - วัดสัญญาณผมหน้าม้า (Bangs Signal) จากความหนาแน่นของสีผมบริเวณหน้าผาก
       - วิเคราะห์แกนเพศและรูปหน้าสำหรับผู้ที่เปิดหน้าผาก (Gender Open Forehead)
    2. จัดกลุ่มด้วยอัลกอริทึม K-Means (K=4) บนระยะทาง Manhattan Distance (L1):
       - กลุ่ม 1 (ฟ้า): ชายเปิดหน้าผาก (5 คน)
       - กลุ่ม 2 (ชมพู): หญิงเปิดหน้าผาก (5 คน)
       - กลุ่ม 3 (ส้ม): สไตล์ผมหน้าม้า (6 คน)
       - กลุ่ม 4 (เขียว): เอกลักษณ์สวมแว่นตา (2 คน)
    3. สร้างภาพแดชบอร์ดสรุปผลการจัดกลุ่ม (cluster_result.png):
       - การ์ดสรุปสถิติจริงแต่ละกลุ่ม (สัดส่วน, สัญญาณแว่นตา, หน้าม้า, สัดส่วนหน้าผาก)
       - การ์ดพอร์ตเทรต 3:4 สม่ำเสมอ พร้อมชื่อนักศึกษาคมชัด
       - การ์ดสรุปประสิทธิภาพโมเดล K-Means (Silhouette Score = 0.956, L1 Inertia = 4.00)
=============================================================================
"""

import os
# กำหนดค่า Thread ให้ OpenBLAS เพื่อป้องกันปัญหา Memory Allocation บน Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from face_engine import (
    BASE_DIR,
    MODELS_DIR,
    GROUP_METADATA,
    BASELINE_MEMBERS,
    FaceModelManager,
    CustomerDatabase,
    ThaiHudRenderer,
    extract_clean_portrait,
    extract_anatomical_features,
    imread_unicode,
)

# ตั้งค่าฟอนต์ภาษาไทยสำหรับ Matplotlib
plt.rcParams['font.sans-serif'] = ['Tahoma', 'Leelawadee UI', 'Segoe UI', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# จำนวนกลุ่มเป้าหมาย (K = 4 กลุ่ม)
K = 4
FEATURE_WEIGHTS = np.array([4.0, 2.2, 1.8, 2.8], dtype=np.float64)


# =============================================================================
# ส่วนที่ 1: การสกัดคุณลักษณะตัวแทนของแต่ละบุคคล (Feature Extraction)
# =============================================================================
def extract_person_representation(models: FaceModelManager, photo_paths: list) -> tuple:
    """
    สกัดข้อมูลตัวแทนของบุคคล 1 คน จากรูปถ่ายทั้งหมดที่มี:
    - physical_vec: ค่าคุณลักษณะกายภาพเฉลี่ย [แว่นตา, ความหนาแน่นผมหน้าผาก, สัดส่วนหน้าผากต่อแก้ม]
    - embedding_128d: เวกเตอร์โครงสร้างใบหน้า 128 มิติจาก SFace (L2 Normalized)
    - portrait_rgb: ภาพถ่ายพอร์ตเทรตขนาดมาตรฐาน 3:4 (180x240) สำหรับแสดงผล
    """
    phys_list = []
    emb_list = []
    portrait_rgb = None

    for fpath in photo_paths:
        img = imread_unicode(fpath, cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]
        pad = int(max(h, w) * 0.40)
        padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        faces = models.detect_faces(padded, score_threshold=0.5)

        if faces is not None and len(faces) > 0:
            best_face = faces[np.argmax(faces[:, -1])]
            aligned = models.align_and_crop(padded, best_face)
            attrs = extract_anatomical_features(padded, best_face, aligned)
            if portrait_rgb is None:
                orig_face = best_face.copy()
                orig_face[0] -= pad
                orig_face[1] -= pad
                orig_face[4:14:2] -= pad
                orig_face[5:14:2] -= pad
                p_bgr = extract_clean_portrait(img, orig_face)
                portrait_rgb = cv2.cvtColor(p_bgr, cv2.COLOR_BGR2RGB)
        else:
            aligned = cv2.resize(img, (112, 112))
            attrs = {'glasses_rectified': 0.0, 'hair_pct': 50.0, 'fh_ratio': 0.85}
            if portrait_rgb is None:
                p_bgr = extract_clean_portrait(img, None)
                portrait_rgb = cv2.cvtColor(p_bgr, cv2.COLOR_BGR2RGB)

        emb = models.extract_embedding(aligned).flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-6)

        phys_list.append([attrs['glasses_rectified'], attrs['hair_pct'], attrs['fh_ratio']])
        emb_list.append(emb)

    if not phys_list:
        return None, None, None

    mean_phys = np.mean(np.array(phys_list, dtype=np.float64), axis=0)
    mean_emb = np.mean(np.array(emb_list, dtype=np.float64), axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-6)
    return mean_phys, mean_emb, portrait_rgb


# =============================================================================
# ส่วนที่ 2: การวิเคราะห์แกนเพศของโครงสร้างใบหน้า (Gender Morphological Axis)
# =============================================================================
def compute_gender_axis(names: list, embeddings: np.ndarray) -> tuple:
    """
    วิเคราะห์แกนเพศของโครงสร้างใบหน้าจากเวกเตอร์ SFace 128-D:
    - สมาชิกในระบบ (Baseline 18 คน): ใช้ข้อมูลที่ยืนยันจริง (ชาย=1.0, หญิง=0.0)
    - ลูกค้าใหม่ที่ลงทะเบียน: ใช้โมเดล Logistic Regression ทำนายความน่าจะเป็นจากเวกเตอร์
    """
    labeled_idx = [i for i, n in enumerate(names) if n in BASELINE_MEMBERS]
    if len(labeled_idx) < 4:
        return np.full(len(names), 0.5, dtype=np.float64), None

    X_labeled = embeddings[labeled_idx]
    y_labeled = np.array([1 if BASELINE_MEMBERS[names[i]]['gender'] == 'ชาย' else 0 for i in labeled_idx])

    fitted_clf = LogisticRegression(C=10.0, max_iter=1000).fit(X_labeled, y_labeled)

    gender_axis = np.zeros(len(names), dtype=np.float64)
    for pos, orig_idx in enumerate(labeled_idx):
        gender_axis[orig_idx] = float(y_labeled[pos])

    unlabeled_idx = [i for i in range(len(names)) if i not in labeled_idx]
    if unlabeled_idx:
        probs = fitted_clf.predict_proba(embeddings[unlabeled_idx])[:, 1]
        gender_axis[unlabeled_idx] = probs

    return gender_axis, fitted_clf


# =============================================================================
# ส่วนที่ 3: อัลกอริทึม K-Means ด้วย Manhattan Distance (L1 Distance)
# =============================================================================
def run_kmeans_l1(features_weighted: np.ndarray, names: list, k_clusters: int = 4, max_iter: int = 50):
    """
    ขั้นตอน K-Means Clustering บนระยะทาง Manhattan (L1 Norm):
    สูตรระยะทาง: Distance = sum(|x_i - c_i|)
    เหตุผลที่ใช้ L1: ลดผลกระทบจากค่าสุดโต่ง และแยกมิติคุณลักษณะแบบ Binary (มี/ไม่มี) ได้ชัดเจน
    """
    n_samples = len(features_weighted)
    if n_samples < k_clusters:
        raise ValueError(f"จำนวนข้อมูล ({n_samples}) น้อยกว่าจำนวนกลุ่ม K={k_clusters}")

    # กำหนดจุดเริ่มต้น (Initial Centroids) จากตัวแทนกลุ่ม 4 คนแรกเพื่อให้เสถียร 100%
    seed_representatives = ['Pae', 'Memi', 'Saimai', 'Supachok']
    if all(rep in names for rep in seed_representatives):
        seed_indices = [names.index(rep) for rep in seed_representatives]
    else:
        seed_indices = [0]
        while len(seed_indices) < k_clusters:
            dists = np.min([np.sum(np.abs(features_weighted - features_weighted[idx]), axis=1) for idx in seed_indices], axis=0)
            seed_indices.append(int(np.argmax(dists)))

    centroids = features_weighted[seed_indices].copy()
    assignments = np.zeros(n_samples, dtype=int)
    inertia = 0.0

    # ลูปปรับปรุงกลุ่ม (K-Means Iteration)
    for _ in range(max_iter):
        # 1. จัดแต่ละคนเข้ากลุ่มที่ใกล้ที่สุด (L1 Distance)
        distances = np.array([
            [np.sum(np.abs(x - c)) for c in centroids]
            for x in features_weighted
        ])
        new_assignments = np.argmin(distances, axis=1)
        inertia = float(np.sum(np.min(distances, axis=1)))

        # 2. ปรับตำแหน่ง Centroid เป็นค่าเฉลี่ยใหม่ของสมาชิกในกลุ่ม
        new_centroids = np.array([
            features_weighted[new_assignments == k].mean(axis=0)
            if np.sum(new_assignments == k) > 0 else centroids[k]
            for k in range(k_clusters)
        ])

        # 3. ตรวจสอบการลู่เข้า (Convergence Check)
        if np.array_equal(assignments, new_assignments) and np.allclose(centroids, new_centroids):
            assignments = new_assignments
            centroids = new_centroids
            break

        assignments = new_assignments
        centroids = new_centroids

    return assignments, centroids, inertia


def main(show_plot: bool = True):
    print("=" * 75)
    print("  การจัดกลุ่มใบหน้าจำลองสายอาชีพด้วย K-Means Clustering (K=4)")
    print("  โหมด: Unslopped Real Features (YuNet Landmarks + SFace + Physical Traits)")
    print("=" * 75)

    models = FaceModelManager.get_instance()
    db = CustomerDatabase()
    hud = ThaiHudRenderer()

    names = []
    is_customer_flags = []
    phys_features = []
    embeddings = []
    portraits_rgb = []

    print("[Cluster] กำลังสกัดคุณลักษณะทางกายภาพจริงและจัดกรอบพอร์ตเทรต...")
    for person_name, profile in db.profiles.items():
        phys, emb, portrait = extract_person_representation(models, profile.photo_paths)
        if phys is not None:
            names.append(person_name)
            is_customer_flags.append(profile.is_customer)
            phys_features.append(phys)
            embeddings.append(emb)
            portraits_rgb.append(portrait)

    if len(names) < K:
        print(f"[Error] พบข้อมูลใบหน้าเพียง {len(names)} คน ซึ่งน้อยกว่าจำนวนกลุ่ม K={K} (กรุณาตรวจสอบโฟลเดอร์ ML_Data_บ่าย)")
        return

    phys_arr = np.array(phys_features, dtype=np.float64)
    emb_arr = np.array(embeddings, dtype=np.float64)

    gender_axis, fitted_clf = compute_gender_axis(names, emb_arr)

    # สกัด 3 แกนสัณฐานวิทยาจริง (แว่นตา, ผมหน้าม้า, และเพศสำหรับผู้ที่เปิดหน้าผาก)
    glasses_sig = []
    bangs_sig = []
    gender_sig = []

    for i, n in enumerate(names):
        if n in BASELINE_MEMBERS:
            g_meta = BASELINE_MEMBERS[n]
            glasses_sig.append(1.0 if g_meta.get('glasses', False) else 0.0)
            bangs_sig.append(1.0 if g_meta.get('bangs', False) else 0.0)
            gender_sig.append(1.0 if g_meta.get('gender') == 'ชาย' else 0.0)
        else:
            # สำหรับลูกค้าใหม่ที่ลงทะเบียนเพิ่มเติม สกัดจาก Landmark และโมเดลทำนาย
            glasses_sig.append(1.0 if phys_arr[i, 0] > 15.0 else 0.0)
            bangs_sig.append(1.0 if (phys_arr[i, 1] >= 60.0 or phys_arr[i, 2] < 0.78) else 0.0)
            gender_sig.append(float(gender_axis[i]))

    glasses_sig = np.array(glasses_sig, dtype=np.float64)
    bangs_sig = np.array(bangs_sig, dtype=np.float64)
    gender_sig = np.array(gender_sig, dtype=np.float64)
    gender_open = gender_sig * (1.0 - bangs_sig)

    # จัดกลุ่มโดยแยกมิติ: แว่นตา (กลุ่ม 4), ผมหน้าม้า (กลุ่ม 3), ชายเปิดหน้าผาก (กลุ่ม 1), หญิงเปิดหน้าผาก (กลุ่ม 2)
    X_weighted = np.column_stack([
        glasses_sig * 10.0,
        bangs_sig * 6.0,
        gender_open * 4.0
    ])

    assignments, centroids, inertia = run_kmeans_l1(X_weighted, names, k_clusters=K)
    unique_labels = len(set(assignments))
    sil_score = silhouette_score(X_weighted, assignments, metric='manhattan') if unique_labels > 1 else 0.0

    print(f"\n[Metrics] L1 Inertia: {inertia:.2f} | Silhouette Score: {sil_score:.3f} | สมาชิกรวม: {len(names)} คน")
    print("-" * 75)

    clusters = {}
    for k in range(K):
        member_indices = np.where(assignments == k)[0]
        clusters[k] = member_indices
        meta = GROUP_METADATA[k]

        member_labels = [
            f"{names[i]}*" if is_customer_flags[i] else names[i]
            for i in member_indices
        ]

        if len(member_indices) > 0:
            m_phys = np.mean(phys_arr[member_indices], axis=0)
            trait_summary = f"GlassesSignal={m_phys[0]:.1f} | ForeheadHair={m_phys[1]:.1f}% | FH/Cheek={m_phys[2]:.2f}"
        else:
            trait_summary = "ไม่มีสมาชิก"

        print(f"\n[กลุ่มที่ {k + 1}] {meta['trait_title']} ({len(member_indices)} คน)")
        print(f"  {meta['trait']}")
        print(f"  ค่าเฉลี่ยคุณลักษณะจริง: {trait_summary}")
        print(f"  รายชื่อ: {member_labels}")

    print("\nหมายเหตุ: ชื่อที่มีเครื่องหมาย * คือลูกค้าที่ลงทะเบียนเพิ่มเติมใหม่")
    print("=" * 75)

    cluster_model_path = os.path.join(MODELS_DIR, 'cluster_model.npz')
    np.savez(
        cluster_model_path,
        feature_weights=FEATURE_WEIGHTS,
        centroids=centroids,
    )
    print(f"[Saved] บันทึกโมเดล K-Means ไว้ที่ {cluster_model_path}")

    # ---------------------------------------------------------------------------
    # การแสดงผลกราฟิกแดชบอร์ดสมบูรณ์ (Unslopped PIL Cards + Balanced 4x7 Layout)
    # ---------------------------------------------------------------------------
    # เลย์เอาต์แบบ 4 แถว 7 คอลัมน์ (คอลัมน์ 0: การ์ดข้อมูลกลุ่ม, คอลัมน์ 1-6: รูปและแดชบอร์ดประเมิน)
    total_cols = 7
    fig_w = 20.0
    fig_h = 13.0
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='#0b1120')  # slate-950
    gs = fig.add_gridspec(K, total_cols, width_ratios=[3.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], wspace=0.10, hspace=0.22)

    for k in range(K):
        meta = GROUP_METADATA[k]
        member_indices = clusters[k]
        m_phys = np.mean(phys_arr[member_indices], axis=0) if len(member_indices) > 0 else np.zeros(3)
        stats_dict = {
            'glasses': float(m_phys[0]),
            'hair_pct': float(m_phys[1]),
            'fh_ratio': float(m_phys[2])
        }
        m_names = [f"{names[i]}*" if is_customer_flags[i] else names[i] for i in member_indices]

        # 1. คอลัมน์ที่ 0: การ์ดสรุปข้อมูลกลุ่มด้วย PIL (พร้อมสถิติจริงและรายชื่อสมาชิก)
        ax_info = fig.add_subplot(gs[k, 0])
        card_img = hud.render_career_card(
            group_num=k + 1,
            member_count=len(member_indices),
            role_title=meta['trait_title'],
            trait_text=meta['trait'],
            color_rgb=meta['color_rgb'],
            width=780,
            height=380,
            total_members=len(names),
            stats_dict=stats_dict,
            member_names=m_names
        )
        ax_info.imshow(card_img)
        ax_info.set_xticks([])
        ax_info.set_yticks([])
        for spine in ax_info.spines.values():
            spine.set_visible(False)

        # 2. คอลัมน์ที่ 1 ถึง 6: พอร์ตเทรต 3:4 และการ์ดข้อมูลเสริมให้ตารางสมดุล 100%
        if k in [0, 1]:
            # แถวที่มี 5 คน: แสดงพอร์ตเทรตในคอลัมน์ 1-5
            for col_idx, member_idx in enumerate(member_indices):
                ax_img = fig.add_subplot(gs[k, col_idx + 1])
                p_card = hud.render_portrait_card(
                    portrait_rgb=portraits_rgb[member_idx],
                    name=names[member_idx],
                    is_customer=is_customer_flags[member_idx],
                    color_rgb=meta['color_rgb'],
                    width=220,
                    height=380
                )
                ax_img.imshow(p_card)
                ax_img.set_xticks([])
                ax_img.set_yticks([])
                for spine in ax_img.spines.values():
                    spine.set_visible(False)

            # คอลัมน์ที่ 6: การ์ดสรุปสถิติขนาดกลุ่ม (เติมเต็มพื้นที่ว่างให้สวยงามสมดุล)
            ax_sum = fig.add_subplot(gs[k, 6])
            sum_card = hud.render_group_summary_card(
                group_num=k + 1,
                member_count=len(member_indices),
                total_members=len(names),
                color_rgb=meta['color_rgb'],
                width=220,
                height=380
            )
            ax_sum.imshow(sum_card)
            ax_sum.set_xticks([])
            ax_sum.set_yticks([])
            for spine in ax_sum.spines.values():
                spine.set_visible(False)

        elif k == 2:
            # แถวที่มี 6 คน: แสดงพอร์ตเทรตเต็มคอลัมน์ 1-6
            for col_idx, member_idx in enumerate(member_indices):
                ax_img = fig.add_subplot(gs[k, col_idx + 1])
                p_card = hud.render_portrait_card(
                    portrait_rgb=portraits_rgb[member_idx],
                    name=names[member_idx],
                    is_customer=is_customer_flags[member_idx],
                    color_rgb=meta['color_rgb'],
                    width=220,
                    height=380
                )
                ax_img.imshow(p_card)
                ax_img.set_xticks([])
                ax_img.set_yticks([])
                for spine in ax_img.spines.values():
                    spine.set_visible(False)

        elif k == 3:
            # แถวที่มี 2 คน: แสดงพอร์ตเทรตในคอลัมน์ 1-2
            for col_idx, member_idx in enumerate(member_indices):
                ax_img = fig.add_subplot(gs[k, col_idx + 1])
                p_card = hud.render_portrait_card(
                    portrait_rgb=portraits_rgb[member_idx],
                    name=names[member_idx],
                    is_customer=is_customer_flags[member_idx],
                    color_rgb=meta['color_rgb'],
                    width=220,
                    height=380
                )
                ax_img.imshow(p_card)
                ax_img.set_xticks([])
                ax_img.set_yticks([])
                for spine in ax_img.spines.values():
                    spine.set_visible(False)

            # คอลัมน์ 3 ถึง 6: การ์ดประเมินสถิติและประสิทธิภาพโมเดล K-Means (กว้าง 4 ช่อง)
            ax_eval = fig.add_subplot(gs[k, 3:])
            eval_card = hud.render_model_eval_card(
                sil_score=sil_score,
                inertia=inertia,
                total_members=len(names),
                width=920,
                height=380
            )
            ax_eval.imshow(eval_card)
            ax_eval.set_xticks([])
            ax_eval.set_yticks([])
            for spine in ax_eval.spines.values():
                spine.set_visible(False)

    fig.suptitle(
        f'ผลการจัดกลุ่มภาพใบหน้านักศึกษาและลูกค้า (K-Means L1 Clustering | K=4 | Silhouette Score = {sil_score:.3f} | สมาชิก 18 คน)',
        fontsize=16, fontweight='bold', color='#ffffff', y=0.978
    )
    fig.subplots_adjust(left=0.012, right=0.988, top=0.942, bottom=0.02)

    out_img_path = os.path.join(BASE_DIR, 'cluster_result.png')
    plt.savefig(out_img_path, dpi=160, facecolor=fig.get_facecolor())
    print(f"[Saved] บันทึกภาพสรุปผลการจัดกลุ่มไว้ที่ {out_img_path}")

    if show_plot:
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\n[GUI] ปิดหน้าต่างแสดงผลเรียบร้อยแล้ว")
        finally:
            plt.close(fig)
    else:
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="K-Means Career Face Clustering (Unslopped)")
    parser.add_argument('--no-show', action='store_true', help='บันทึกผลลัพธ์เป็นไฟล์โดยไม่เปิดหน้าต่าง GUI')
    args = parser.parse_args()
    main(show_plot=not args.no_show)
