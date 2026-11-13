"""
=============================================================================
หน้าหลักสำหรับผู้ใช้งานและลูกค้า (Unified Launcher)
ไฟล์: main.py
คำอธิบาย:
    หน้าต่างเมนูหลัก (Desktop UI) เพื่อให้ลูกค้าและผู้ใช้งานเปิดระบบได้ง่ายในคลิกเดียว:
    1. สแกนใบหน้าเข้าใช้งาน (FaceRecognize.py)
    2. ลงทะเบียนลูกค้าใหม่ (FaceTrain.py)
    3. วิเคราะห์จัดกลุ่มสายอาชีพ K-Means (FaceCluster.py)
=============================================================================
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_module(script_name: str, root: tk.Tk):
    script_path = os.path.join(BASE_DIR, script_name)
    if not os.path.exists(script_path):
        messagebox.showerror("ไม่พบไฟล์", f"ไม่พบไฟล์ {script_name}")
        return
    root.withdraw()
    try:
        subprocess.run([sys.executable, script_path], cwd=BASE_DIR)
    finally:
        root.deiconify()


def main():
    root = tk.Tk()
    root.title("ระบบสแกนและจัดกลุ่มใบหน้าลูกค้าอัจฉริยะ (Smart Face System)")
    root.geometry("480x390")
    root.configure(bg="#1e272e")
    root.resizable(False, False)

    title_lbl = tk.Label(
        root,
        text="ระบบจัดกลุ่มภาพใบหน้า (K-Means Clustering)",
        font=("Tahoma", 16, "bold"),
        fg="#ffffff",
        bg="#1e272e",
        pady=14
    )
    title_lbl.pack()

    sub_lbl = tk.Label(
        root,
        text="วิเคราะห์และจัดกลุ่มอัตโนมัติ (K=4) จากคุณลักษณะทางกายภาพจริง",
        font=("Tahoma", 10),
        fg="#b2bec3",
        bg="#1e272e"
    )
    sub_lbl.pack(pady=(0, 20))

    btn_style = {
        "font": ("Tahoma", 13, "bold"),
        "fg": "#ffffff",
        "width": 34,
        "height": 2,
        "bd": 0,
        "cursor": "hand2",
        "activeforeground": "#ffffff",
    }

    tk.Button(
        root,
        text="📊  เริ่มการจัดกลุ่มใบหน้า K-Means (K=4)",
        bg="#e67e22",
        activebackground="#d35400",
        command=lambda: run_module("FaceCluster.py", root),
        **btn_style
    ).pack(pady=20)

    tk.Label(
        root,
        text="คลิกเลือกโหมดที่ต้องการ • กด [Q] ในหน้ากล้องเพื่อกลับสู่เมนูนี้",
        font=("Tahoma", 9),
        fg="#7f8c8d",
        bg="#1e272e",
        pady=16
    ).pack(side=tk.BOTTOM)

    root.mainloop()


if __name__ == '__main__':
    main()

