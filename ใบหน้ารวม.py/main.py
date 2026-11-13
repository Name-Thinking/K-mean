"""
=============================================================================
โครงงาน: การจัดกลุ่มภาพใบหน้า (K-Means Clustering: K=4)
วิชา: Machine Learning
ไฟล์: main.py
คำอธิบาย:
    หน้าต่างเมนูหลัก (Desktop UI) สำหรับเปิดใช้งานระบบจัดกลุ่มภาพใบหน้า:
    - สกัดคุณลักษณะทางกายภาพจริง (แว่นตา, ผมหน้าม้า, โครงสร้างเพศ) ด้วย YuNet + SFace
    - จัดกลุ่มด้วย K-Means Clustering (K=4, Manhattan Distance L1)
    - แสดงภาพพอร์ตเทรต 3:4 สม่ำเสมอ และบันทึกผลลง cluster_result.png
=============================================================================
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULT_PNG = os.path.join(BASE_DIR, 'cluster_result.png')
if not os.path.exists(RESULT_PNG):
    RESULT_PNG = os.path.join(PARENT_DIR, 'cluster_result.png')


def run_cluster_gui(root: tk.Tk):
    script_path = os.path.join(BASE_DIR, "FaceCluster.py")
    if not os.path.exists(script_path):
        messagebox.showerror("ไม่พบไฟล์", f"ไม่พบไฟล์ {script_path}")
        return
    root.withdraw()
    try:
        subprocess.run([sys.executable, script_path], cwd=BASE_DIR)
    finally:
        root.deiconify()


def run_cluster_headless(root: tk.Tk):
    script_path = os.path.join(BASE_DIR, "FaceCluster.py")
    if not os.path.exists(script_path):
        messagebox.showerror("ไม่พบไฟล์", f"ไม่พบไฟล์ {script_path}")
        return
    try:
        res = subprocess.run([sys.executable, script_path, "--no-show"], cwd=BASE_DIR, capture_output=True, text=True)
        if res.returncode == 0:
            messagebox.showinfo("สำเร็จ", "ประมวลผลและบันทึกภาพ cluster_result.png เรียบร้อยแล้ว!")
        else:
            messagebox.showerror("เกิดข้อผิดพลาด", res.stderr[:400] if res.stderr else "การประมวลผลล้มเหลว")
    except Exception as e:
        messagebox.showerror("เกิดข้อผิดพลาด", str(e))


def open_result_image():
    target = RESULT_PNG
    if not os.path.exists(target):
        target = os.path.join(BASE_DIR, 'cluster_result.png')
    if os.path.exists(target):
        if sys.platform == 'win32':
            os.startfile(target)
        else:
            subprocess.run(['xdg-open', target])
    else:
        messagebox.showwarning("ยังไม่มีไฟล์", "กรุณากดจัดกลุ่มก่อนเพื่อสร้างไฟล์ cluster_result.png")


def main():
    root = tk.Tk()
    root.title("ระบบจัดกลุ่มภาพใบหน้า (K-Means Clustering: K=4)")
    root.geometry("520x460")
    root.configure(bg="#1e272e")
    root.resizable(False, False)

    # หัวข้อ
    tk.Label(
        root,
        text="ระบบจัดกลุ่มภาพใบหน้า (K-Means Clustering)",
        font=("Tahoma", 16, "bold"),
        fg="#ffffff",
        bg="#1e272e",
        pady=12
    ).pack()

    tk.Label(
        root,
        text="วิเคราะห์ 4 กลุ่มตามคุณลักษณะกายภาพจริง 100% (K=4, L1 Manhattan)",
        font=("Tahoma", 10),
        fg="#b2bec3",
        bg="#1e272e"
    ).pack(pady=(0, 16))

    # กล่องสรุปกลุ่ม 4 กลุ่ม
    summary_frame = tk.Frame(root, bg="#2f3640", padx=16, pady=10, relief=tk.GROOVE, bd=1)
    summary_frame.pack(fill=tk.X, padx=28, pady=(0, 20))

    groups_info = [
        ("กลุ่มที่ 1 (ฟ้า): ชายเปิดหน้าผาก (5 คน)", "#3498db"),
        ("กลุ่มที่ 2 (ชมพู): หญิงเปิดหน้าผาก (5 คน)", "#e84393"),
        ("กลุ่มที่ 3 (ส้ม): สไตล์ผมหน้าม้า (6 คน)", "#e67e22"),
        ("กลุ่มที่ 4 (เขียว): เอกลักษณ์สวมแว่นตา (2 คน)", "#2ecc71"),
    ]
    for text, color in groups_info:
        tk.Label(
            summary_frame,
            text=f"• {text}",
            font=("Tahoma", 9, "bold"),
            fg=color,
            bg="#2f3640",
            anchor="w"
        ).pack(fill=tk.X, pady=2)

    btn_style = {
        "font": ("Tahoma", 11, "bold"),
        "fg": "#ffffff",
        "width": 38,
        "height": 2,
        "bd": 0,
        "cursor": "hand2",
        "activeforeground": "#ffffff",
    }

    # ปุ่ม 1: เปิดหน้าต่างกราฟิกเต็ม
    tk.Button(
        root,
        text="📊  เริ่มจัดกลุ่มและแสดงหน้าต่างกราฟิก (GUI Interactive)",
        bg="#e67e22",
        activebackground="#d35400",
        command=lambda: run_cluster_gui(root),
        **btn_style
    ).pack(pady=6)

    # ปุ่ม 2: ประมวลผลและเซฟภาพ
    tk.Button(
        root,
        text="⚡  ประมวลผลด่วนและบันทึกภาพ (Export Headless)",
        bg="#3498db",
        activebackground="#2980b9",
        command=lambda: run_cluster_headless(root),
        **btn_style
    ).pack(pady=6)

    # ปุ่ม 3: เปิดดูรูปภาพผลลัพธ์
    tk.Button(
        root,
        text="🖼️  เปิดดูไฟล์ผลลัพธ์ (cluster_result.png)",
        bg="#2ecc71",
        activebackground="#27ae60",
        command=open_result_image,
        **btn_style
    ).pack(pady=6)

    tk.Label(
        root,
        text="ภาควิชาวิศวกรรมและเทคโนโลยี • โมเดล YuNet + SFace Deep Learning",
        font=("Tahoma", 8),
        fg="#7f8c8d",
        bg="#1e272e",
        pady=10
    ).pack(side=tk.BOTTOM)

    root.mainloop()


if __name__ == '__main__':
    main()
