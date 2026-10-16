# Face Clustering with K-Means (K=4)

โปรเจกต์จัดกลุ่มภาพใบหน้าโดยใช้อัลกอริทึม **K-Means Clustering** ร่วมกับ **Manhattan Distance (L1 Norm)**

## จุดเด่นของโปรเจกต์
- **Feature Extraction (10 Features):** สกัดคุณลักษณะเด่นของใบหน้า เช่น คอนทราสต์กรอบแว่นตา, ความสว่างหน้าผาก (ตรวจจับผมหน้าม้า), โครงสร้างคิ้วและแนวกราม, โทนสีผิว, ความสมมาตร
- **Feature Normalization:** ทำ Z-score normalization (Mean=0, Std=1)
- **K-Means with Manhattan Distance:** จัดกลุ่มข้อมูลภาพออกเป็น 4 กลุ่มหลัก:
  1. **Group 1: ผู้ชาย (Men)** - Pae, Jang, Ball, Mart, VA, North, Endu, Champ
  2. **Group 2: ผู้หญิงเปิดหน้าผาก (Women - Open Forehead)** - Memi, Pan, First, Nod, Ann, DD
  3. **Group 3: ผู้หญิงผมหน้าม้า (Women - Bangs)** - Saimai, Earn
  4. **Group 4: บุคคลสวมแว่นตา (Glasses)** - Manow, Supachok
- **Visual Display:** แสดงผลลัพธ์ผ่าน Terminal และตารางภาพใบหน้าด้วย Matplotlib พร้อมสีกรอบและป้ายชื่อแยกตามกลุ่ม

## การติดตั้งและใช้งาน
```bash
pip install opencv-python numpy matplotlib
python FaceCluster.py
```
