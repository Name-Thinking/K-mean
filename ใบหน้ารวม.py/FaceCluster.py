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

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut
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

plt.rcParams['font.sans-serif'] = ['Tahoma', 'Leelawadee UI', 'Segoe UI', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

K = 4
FEATURE_WEIGHTS = np.array([4.0, 2.2, 1.8, 2.8], dtype=np.float64)


def extract_person_representation(models: FaceModelManager, photo_paths: list) -> tuple:
    """
    สกัดข้อมูลตัวแทนของบุคคล 1 คนจากภาพทั้งหมดของคนนั้น:
      - physical_vec: [glasses_rectified, hair_pct, fh_ratio]
      - embedding_128d: เวกเตอร์ L2-normalized เฉลี่ย
      - portrait_rgb: ภาพพอร์ตเทรตขนาดมาตรฐาน 3:4 (180x240) สำหรับแสดงในตารางอย่างสวยงาม
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
                # แปลงพิกัดกลับเป็นรูปต้นฉบับ (ไม่ให้ติดขอบ BORDER_REPLICATE!)
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


def compute_gender_axis(names: list, embeddings: np.ndarray) -> tuple:
    """
    เรียนรู้แกนสัณฐานวิทยาของโครงสร้างใบหน้า (Gender Morphology Axis) จาก SFace 128-D
    โดยใช้ Cross-Validation แบบ Out-of-Fold (LeaveOneOut) สำหรับชุดข้อมูลตั้งต้นเพื่อป้องกัน Overfitting
    และใช้โมเดลที่ฝึกแล้วทำนายให้ลูกค้าใหม่ที่ลงทะเบียนเพิ่มเติมอย่างแท้จริง
    """
    labeled_idx = [i for i, n in enumerate(names) if n in BASELINE_MEMBERS]
    if len(labeled_idx) < 4:
        return np.full(len(names), 0.5, dtype=np.float64), None

    X_labeled = embeddings[labeled_idx]
    y_labeled = np.array([1 if BASELINE_MEMBERS[names[i]]['gender'] == 'ชาย' else 0 for i in labeled_idx])

    clf_base = LogisticRegression(C=10.0, max_iter=1000)
    oof_scores = cross_val_predict(
        clf_base, X_labeled, y_labeled, cv=LeaveOneOut(), method='decision_function'
    )

    fitted_clf = LogisticRegression(C=10.0, max_iter=1000).fit(X_labeled, y_labeled)

    all_scores = np.zeros(len(names), dtype=np.float64)
    for pos, orig_idx in enumerate(labeled_idx):
        all_scores[orig_idx] = oof_scores[pos]

    unlabeled_idx = [i for i in range(len(names)) if i not in labeled_idx]
    if unlabeled_idx:
        all_scores[unlabeled_idx] = fitted_clf.decision_function(embeddings[unlabeled_idx])

    gender_axis = 1.0 / (1.0 + np.exp(-all_scores * 2.5))
    return gender_axis, fitted_clf


def run_kmeans_l1(features_weighted: np.ndarray, names: list, k_clusters: int = 4, max_iter: int = 50):
    n_samples = len(features_weighted)
    if n_samples < k_clusters:
        raise ValueError(f"จำนวนข้อมูล ({n_samples}) น้อยกว่าจำนวนกลุ่ม K={k_clusters}")

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

    for _ in range(max_iter):
        distances = np.array([
            [np.sum(np.abs(x - c)) for c in centroids]
            for x in features_weighted
        ])
        new_assignments = np.argmin(distances, axis=1)
        inertia = float(np.sum(np.min(distances, axis=1)))

        new_centroids = np.array([
            features_weighted[new_assignments == k].mean(axis=0)
            if np.sum(new_assignments == k) > 0 else centroids[k]
            for k in range(k_clusters)
        ])

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

    gender_axis, _ = compute_gender_axis(names, emb_arr)
    X_raw = np.hstack([phys_arr, gender_axis.reshape(-1, 1)])

    feat_mean = np.mean(X_raw, axis=0)
    feat_std = np.std(X_raw, axis=0)
    feat_std[feat_std == 0] = 1.0
    X_weighted = ((X_raw - feat_mean) / feat_std) * FEATURE_WEIGHTS

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

        print(f"\n[กลุ่มที่ {k + 1}] {meta['role']} ({len(member_indices)} คน)")
        print(f"  {meta['trait']}")
        print(f"  ค่าเฉลี่ยคุณลักษณะจริง: {trait_summary}")
        print(f"  รายชื่อ: {member_labels}")

    print("\nหมายเหตุ: ชื่อที่มีเครื่องหมาย * คือลูกค้าที่ลงทะเบียนเพิ่มเติมใหม่")
    print("=" * 75)

    cluster_model_path = os.path.join(MODELS_DIR, 'cluster_model.npz')
    np.savez(
        cluster_model_path,
        feat_mean=feat_mean,
        feat_std=feat_std,
        feature_weights=FEATURE_WEIGHTS,
        centroids=centroids,
    )
    print(f"[Saved] บันทึกโมเดล K-Means ไว้ที่ {cluster_model_path}")

    # ---------------------------------------------------------------------------
    # การแสดงผลกราฟิกแบบสมบูรณ์ (PIL Crisp Thai Cards + Uniform 3:4 Portraits)
    # ---------------------------------------------------------------------------
    max_cols = max((len(clusters[k]) for k in range(K)), default=1)
    max_cols = max(max_cols, 1)
    total_cols = max_cols + 1

    fig_w = max(16.0, 3.8 + max_cols * 2.05)
    fig = plt.figure(figsize=(fig_w, 11.2), facecolor='#1e272e')
    gs = fig.add_gridspec(K, total_cols, width_ratios=[3.2] + [1.0] * max_cols, wspace=0.14, hspace=0.28)

    for k in range(K):
        meta = GROUP_METADATA[k]
        member_indices = clusters[k]
        color_hex = '#{:02x}{:02x}{:02x}'.format(*meta['color_rgb'])

        # คอลัมน์ที่ 0: การ์ดข้อความเรนเดอร์ด้วย PIL (ไม่มีข้อความล้นกรอบหรือทับรูป 100%)
        ax_info = fig.add_subplot(gs[k, 0])
        card_img = hud.render_career_card(
            group_num=k + 1,
            member_count=len(member_indices),
            role_title=meta['role'],
            trait_text=meta['trait'],
            color_rgb=meta['color_rgb'],
            width=760,
            height=380
        )
        ax_info.imshow(card_img)
        ax_info.set_xticks([])
        ax_info.set_yticks([])
        for spine in ax_info.spines.values():
            spine.set_visible(False)

        # คอลัมน์ถัดไป: ภาพพอร์ตเทรตมาตรฐาน 3:4 ที่เห็นใบหน้าและผมสมบูรณ์สม่ำเสมอ
        for col_idx, member_idx in enumerate(member_indices):
            ax_img = fig.add_subplot(gs[k, col_idx + 1])
            ax_img.imshow(portraits_rgb[member_idx])
            ax_img.set_xticks([])
            ax_img.set_yticks([])

            for spine in ax_img.spines.values():
                spine.set_color(color_hex)
                spine.set_linewidth(2.8)

            tag = " [ลูกค้า]" if is_customer_flags[member_idx] else ""
            ax_img.set_title(
                f"{names[member_idx]}{tag}",
                fontsize=10.5, fontweight='bold', color='white', pad=6
            )

    fig.suptitle(
        f'ผลการจัดกลุ่มภาพใบหน้านักศึกษาและลูกค้าจำลองสายอาชีพ (K-Means L1 | K=4 | Silhouette={sil_score:.3f})',
        fontsize=15.5, fontweight='bold', color='white', y=0.965
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.90, bottom=0.03)

    out_img_path = os.path.join(BASE_DIR, 'cluster_result.png')
    plt.savefig(out_img_path, dpi=160, facecolor=fig.get_facecolor())
    print(f"[Saved] บันทึกภาพสรุปผลการจัดกลุ่มไว้ที่ {out_img_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="K-Means Career Face Clustering (Unslopped)")
    parser.add_argument('--no-show', action='store_true', help='บันทึกผลลัพธ์เป็นไฟล์โดยไม่เปิดหน้าต่าง GUI')
    args = parser.parse_args()
    main(show_plot=not args.no_show)
