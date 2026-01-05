#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Four-camera two-by-two Stereo Capture
- Pairs: [12] (Cam1-Cam2) and [34] (Cam3-Cam4)
- Mode switch: [1]=pair 1-2, [3]=pair 3-4
- Capture: [SPACE] (only saves when BOTH cams in the current pair found the checkerboard)
- Quit:    [q] or [ESC]
- Saves RAW PNG only (no overlay), filenames: left_{N}.png / right_{N}.png
- Colored checkerboard overlay is for preview only
"""

import cv2
import time
import os
import re
import numpy as np
from pathlib import Path
from typing import Tuple, List, Union

# =========================
# User Config
# =========================
# 4 台相機來源（依實機調整；可用 /dev/videoX）
#四台相機來源會依照USB插入順序編號
CAM_SOURCES: List[Union[int, str]] = [0,2,4,6]

# 目標解析度（需和主程式解析度一致）
TARGET_SIZE = (1920, 1080)   # (w, h)

# 棋盤格內角點 (cols, rows)
CHECKERBOARD_SIZE = (4, 3)

# === 存檔路徑（pair12 與 pair34）===
# 請將路徑改為你想存放的位置
# pair 1-2：
SAVE_PATH_12_L = Path("/home/an/tm_ws/light/pair12/img_l")  # Cam1 -> left
SAVE_PATH_12_R = Path("/home/an/tm_ws/light/pair12/img_r")  # Cam2 -> right
# pair 3-4：
SAVE_PATH_34_L = Path("/home/an/tm_ws/light/pair34/img_l")  # Cam3 -> left
SAVE_PATH_34_R = Path("/home/an/tm_ws/light/pair34/img_r")  # Cam4 -> right

# 顯示視窗大小（只影響顯示，不影響存檔）
DISPLAY_SIZE = (1280, 720)

# 角點偵測與細化
FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
SUBPIX_WIN = (11, 11)
SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)

LINE_THICK = 2
# =========================

def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def open_cams(sources: List[Union[int, str]], size: Tuple[int,int]) -> List[cv2.VideoCapture]:
    caps = []
    for i, src in enumerate(sources):
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {i+1} (source={src})")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        caps.append(cap)
  
    time.sleep(0.5)
    return caps

def read_and_resize(cap: cv2.VideoCapture, size: Tuple[int,int]) -> np.ndarray:
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to grab frame")
    h, w = frame.shape[:2]
    if (w, h) != size:
        interp = cv2.INTER_AREA if (w > size[0] and h > size[1]) else cv2.INTER_LINEAR
        frame = cv2.resize(frame, size, interpolation=interp)
    return frame

def detect_corners(gray: np.ndarray, board_size: Tuple[int,int]):
    ret, corners = cv2.findChessboardCorners(gray, board_size, FIND_FLAGS)
    if not ret:
        return False, None
    corners = cv2.cornerSubPix(gray, corners, SUBPIX_WIN, (-1, -1), SUBPIX_CRIT)
    return True, corners

def draw_checkerboard_overlay(bgr: np.ndarray, corners: np.ndarray, board_size: Tuple[int,int]) -> None:
    cols, rows = board_size
    pts = corners.reshape(-1, 2)

    row_colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 165, 255), (255, 255, 0)]
    for r in range(rows):
        color = row_colors[r % len(row_colors)]
        for c in range(cols - 1):
            i1 = r * cols + c
            i2 = r * cols + c + 1
            p1 = tuple(np.int32(pts[i1]))
            p2 = tuple(np.int32(pts[i2]))
            cv2.line(bgr, p1, p2, color, LINE_THICK, cv2.LINE_AA)

    col_colors = [(0, 255, 0), (0, 128, 255), (128, 0, 255), (255, 128, 0), (0, 255, 128)]
    for c in range(cols):
        color = col_colors[c % len(col_colors)]
        for r in range(rows - 1):
            i1 = r * cols + c
            i2 = (r + 1) * cols + c
            p1 = tuple(np.int32(pts[i1]))
            p2 = tuple(np.int32(pts[i2]))
            cv2.line(bgr, p1, p2, color, LINE_THICK, cv2.LINE_AA)

    cv2.drawChessboardCorners(bgr, board_size, corners, True)

def put_hud(img: np.ndarray, text: str, ok: bool):
    h, w = img.shape[:2]
    bg = (0, 0, 0)
    fg = (0, 255, 0) if ok else (0, 0, 255)
    cv2.rectangle(img, (0, 0), (w, 34), bg, -1)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fg, 2, cv2.LINE_AA)

def next_counter_in(folder: Path, prefix: str) -> int:

    folder.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.png$", re.IGNORECASE)
    max_n = -1
    for name in os.listdir(folder):
        m = pattern.match(name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1

def main():

    ensure_dirs(SAVE_PATH_12_L, SAVE_PATH_12_R, SAVE_PATH_34_L, SAVE_PATH_34_R)


    print("[INFO] Opening cameras:", CAM_SOURCES)
    caps = open_cams(CAM_SOURCES, TARGET_SIZE)  


    counter_12 = max(next_counter_in(SAVE_PATH_12_L, "left"),
                     next_counter_in(SAVE_PATH_12_R, "right"))
    counter_34 = max(next_counter_in(SAVE_PATH_34_L, "left"),
                     next_counter_in(SAVE_PATH_34_R, "right"))

    mode = '12'  
    print("[INFO] Start in mode: 1-2 (press '1' or '3' to switch)")

 
    for wn in ["Cam1", "Cam2", "Cam3", "Cam4"]:
        cv2.namedWindow(wn, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(wn, DISPLAY_SIZE[0]//2, DISPLAY_SIZE[1]//2)

    while True:
 
        frames, grays = [], []
        for cap in caps:
            f = read_and_resize(cap, TARGET_SIZE)
            frames.append(f)
            grays.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))


        found, corners, overlays = [], [], []
        for i in range(4):
            ok, pts = detect_corners(grays[i], CHECKERBOARD_SIZE)
            found.append(ok)
            corners.append(pts)
            vis = frames[i].copy()
            if ok:
                draw_checkerboard_overlay(vis, pts, CHECKERBOARD_SIZE)
            overlays.append(vis)


        tip = f"[Mode {mode}] SPACE=capture  1=pair12  3=pair34  q/ESC=quit"
        put_hud(overlays[0], f"{tip} | Cam1 {'OK' if found[0] else '---'}", found[0])
        put_hud(overlays[1], f"{tip} | Cam2 {'OK' if found[1] else '---'}", found[1])
        put_hud(overlays[2], f"{tip} | Cam3 {'OK' if found[2] else '---'}", found[2])
        put_hud(overlays[3], f"{tip} | Cam4 {'OK' if found[3] else '---'}", found[3])


        cv2.imshow("Cam1", overlays[0])
        cv2.imshow("Cam2", overlays[1])
        cv2.imshow("Cam3", overlays[2])
        cv2.imshow("Cam4", overlays[3])

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): 
            print("程式結束。")
            break
        elif key == ord('1'):
            mode = '12'
            print("[MODE] Switched to pair 1-2")
        elif key == ord('3'):
            mode = '34'
            print("[MODE] Switched to pair 3-4")
        elif key == 32:  
            if mode == '12':
                if found[0] and found[1]:
                    n = counter_12
                    pL = SAVE_PATH_12_L / f"left_{n}.png"   # Cam1
                    pR = SAVE_PATH_12_R / f"right_{n}.png"  # Cam2
                    cv2.imwrite(str(pL), frames[0])
                    cv2.imwrite(str(pR), frames[1])
                    counter_12 += 1
                    print(f"成功儲存 (pair12)! [{n}] -> {pL} 和 {pR}")
                else:
                    miss = []
                    if not found[0]: miss.append("Cam1")
                    if not found[1]: miss.append("Cam2")
                    print(f"錯誤: 拍照失敗（pair12）。未偵測到棋盤格：{', '.join(miss)}")
            else:  # mode == '34'
                if found[2] and found[3]:
                    n = counter_34
                    pL = SAVE_PATH_34_L / f"left_{n}.png"   # Cam3
                    pR = SAVE_PATH_34_R / f"right_{n}.png"  # Cam4
                    cv2.imwrite(str(pL), frames[2])
                    cv2.imwrite(str(pR), frames[3])
                    counter_34 += 1
                    print(f"成功儲存 (pair34)! [{n}] -> {pL} 和 {pR}")
                else:
                    miss = []
                    if not found[2]: miss.append("Cam3")
                    if not found[3]: miss.append("Cam4")
                    print(f"錯誤: 拍照失敗（pair34）。未偵測到棋盤格：{', '.join(miss)}")

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
