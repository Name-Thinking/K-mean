import os
import sys
import shutil
import unittest

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
    get_baseline_data_folder,
    imread_unicode,
    CUSTOMERS_DIR,
    PORTRAIT_SIZE,
)
from FaceCluster import extract_person_representation, compute_gender_axis, run_kmeans_l1, FEATURE_WEIGHTS


class TestFaceSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = FaceModelManager.get_instance()
        cls.db = CustomerDatabase()
        cls.baseline_dir = get_baseline_data_folder()

    def test_01_baseline_database_loads_18_profiles(self):
        self.assertGreaterEqual(len(self.db.profiles), 18)
        for name, prof in self.db.profiles.items():
            self.assertGreater(len(prof.embeddings), 0, f"{name} has no embeddings")

    def test_02_offcenter_and_dimmed_recognition_accuracy(self):
        correct = 0
        total = 0
        for fname in sorted(os.listdir(self.baseline_dir)):
            if not fname.endswith('.jpg'):
                continue
            true_name = fname.replace('.jpg', '').split('_')[1]
            if true_name not in self.db.profiles:
                continue

            img = imread_unicode(os.path.join(self.baseline_dir, fname))
            h, w = img.shape[:2]
            scale = 180.0 / h
            resized = cv2.resize(img, (int(w * scale), int(h * scale)))
            darkened = np.clip(resized.astype(np.float32) * 0.85, 0, 255).astype(np.uint8)
            rh, rw = darkened.shape[:2]

            canvas = np.full((480, 640, 3), 80, dtype=np.uint8)
            canvas[220:220 + rh, 50:50 + rw] = darkened

            faces = self.models.detect_faces(canvas, score_threshold=0.5)
            self.assertIsNotNone(faces, f"No face detected for {true_name}")
            self.assertGreater(len(faces), 0)

            res = self.db.recognize_face(canvas, faces[0])
            total += 1
            if res.matched and res.name == true_name:
                correct += 1

        self.assertEqual(total, 18)
        self.assertEqual(correct, 18, f"Expected 18/18 accuracy, got {correct}/{total}")

    def test_03_unknown_stranger_rejection(self):
        temp_db = CustomerDatabase()
        pae_profile = temp_db.profiles.pop('Pae')
        temp_db.all_embeddings = []
        temp_db.embedding_owners = []
        for owner, prof in temp_db.profiles.items():
            for emb in prof.embeddings:
                temp_db.all_embeddings.append(emb)
                temp_db.embedding_owners.append(owner)

        pae_img = imread_unicode(pae_profile.photo_paths[0])
        canvas = np.full((480, 640, 3), 100, dtype=np.uint8)
        ph, pw = pae_img.shape[:2]
        canvas[100:100 + ph, 400:400 + pw] = pae_img

        faces = self.models.detect_faces(canvas, score_threshold=0.5)
        self.assertIsNotNone(faces)
        res = temp_db.recognize_face(canvas, faces[0])
        self.assertFalse(res.matched, "Unknown stranger should NOT be matched!")
        self.assertEqual(res.name, "Unknown")

    def test_04_quality_gates_blur_dark_and_small(self):
        blank = np.full((480, 640, 3), 128, dtype=np.uint8)
        small_face = np.array([100, 100, 40, 45, 115, 115, 125, 115, 120, 125, 115, 135, 125, 135, 0.9], dtype=np.float32)
        q_small = assess_face_quality(blank, small_face)
        self.assertFalse(q_small.is_valid)
        self.assertTrue(q_small.is_too_small)

        dark_img = np.full((480, 640, 3), 10, dtype=np.uint8)
        normal_face = np.array([100, 100, 120, 150, 140, 150, 180, 150, 160, 175, 145, 210, 175, 210, 0.9], dtype=np.float32)
        q_dark = assess_face_quality(dark_img, normal_face)
        self.assertFalse(q_dark.is_valid)
        self.assertTrue(q_dark.is_poor_lighting)

        blur_img = np.full((480, 640, 3), 128, dtype=np.uint8)
        q_blur = assess_face_quality(blur_img, normal_face)
        self.assertFalse(q_blur.is_valid)
        self.assertTrue(q_blur.is_blurry)

    def test_05_thai_customer_enrollment_and_reload(self):
        test_name = "__ทดสอบ_CustomerUnit__"
        cust_path = os.path.join(CUSTOMERS_DIR, test_name)
        profiles_json = os.path.join(CUSTOMERS_DIR, 'profiles.json')
        backup_json = None
        if os.path.exists(profiles_json):
            with open(profiles_json, 'r', encoding='utf-8') as f:
                backup_json = f.read()

        try:
            pae_img = imread_unicode(self.db.profiles['Pae'].photo_paths[0])
            pad = 60
            padded = cv2.copyMakeBorder(pae_img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            faces = self.models.detect_faces(padded, 0.5)
            aligned = self.models.align_and_crop(padded, faces[0])
            portrait = extract_clean_portrait(pae_img, None)

            temp_db = CustomerDatabase()
            temp_db.register_new_customer(test_name, [portrait, portrait], aligned_faces=[aligned, aligned], group_id=0, gender='ชาย')

            reloaded_db = CustomerDatabase()
            self.assertIn(test_name, reloaded_db.profiles)
            reloaded_db.profiles.pop('Pae', None)
            reloaded_db.all_embeddings = []
            reloaded_db.embedding_owners = []
            for owner, prof in reloaded_db.profiles.items():
                for emb in prof.embeddings:
                    reloaded_db.all_embeddings.append(emb)
                    reloaded_db.embedding_owners.append(owner)

            res = reloaded_db.recognize_face(padded, faces[0])
            self.assertTrue(res.matched)
            self.assertEqual(res.name, test_name)
        finally:
            shutil.rmtree(cust_path, ignore_errors=True)
            if backup_json is not None:
                with open(profiles_json, 'w', encoding='utf-8') as f:
                    f.write(backup_json)
            elif os.path.exists(profiles_json):
                os.remove(profiles_json)

    def test_06_portrait_framing_and_glasses_separability(self):
        sup_phys, _, sup_port = extract_person_representation(self.models, self.db.profiles['Supachok'].photo_paths)
        man_phys, _, man_port = extract_person_representation(self.models, self.db.profiles['Manow'].photo_paths)
        mem_phys, _, mem_port = extract_person_representation(self.models, self.db.profiles['Memi'].photo_paths)
        ann_phys, _, ann_port = extract_person_representation(self.models, self.db.profiles['Ann'].photo_paths)

        for p in (sup_port, man_port, mem_port, ann_port):
            self.assertEqual(p.shape, (PORTRAIT_SIZE[1], PORTRAIT_SIZE[0], 3))

        self.assertGreater(sup_phys[0], 15.0)
        self.assertGreater(man_phys[0], 15.0)
        self.assertEqual(mem_phys[0], 0.0)
        self.assertEqual(ann_phys[0], 0.0)

    def test_07_kmeans_l1_and_card_renderer(self):
        names = list(self.db.profiles.keys())
        phys_list, emb_list = [], []
        for n in names:
            p, e, _ = extract_person_representation(self.models, self.db.profiles[n].photo_paths)
            phys_list.append(p)
            emb_list.append(e)

        phys_arr = np.array(phys_list, dtype=np.float64)
        emb_arr = np.array(emb_list, dtype=np.float64)
        gender_axis, _ = compute_gender_axis(names, emb_arr)
        X_raw = np.hstack([phys_arr, gender_axis.reshape(-1, 1)])
        X_weighted = ((X_raw - X_raw.mean(axis=0)) / (X_raw.std(axis=0) + 1e-9)) * FEATURE_WEIGHTS

        assign, centroids, inertia = run_kmeans_l1(X_weighted, names, k_clusters=4)
        self.assertEqual(len(assign), len(names))
        self.assertEqual(centroids.shape, (4, 4))
        self.assertGreater(inertia, 0.0)

        hud = ThaiHudRenderer()
        card = hud.render_career_card(1, 4, "ทดสอบ (Test)", "รายละเอียดทดสอบ", (52, 152, 219))
        self.assertEqual(card.shape, (380, 720, 3))


if __name__ == '__main__':
    unittest.main(verbosity=2)

