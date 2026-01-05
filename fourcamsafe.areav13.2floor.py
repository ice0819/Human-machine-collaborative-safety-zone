#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, re, cv2, time, glob, math, threading, signal
import numpy as np
import pybullet as p
import pybullet_data
from dataclasses import dataclass

# ====== YOLO / Torch ======
import torch
from ultralytics import YOLO

# ====== ReID (OSNet) ======
try:
    import torchreid
except ImportError:
    torchreid = None
    print("[WARN] torchreid 未安裝，將停用 ReID（stereo 配對時只有幾何與投影先驗）")

# ====== ROS2 ======
import rclpy
from rclpy.node import Node
from tm_msgs.srv import SendScript, SetEvent
from tm_msgs.msg import FeedbackState

# ====== 指派 ======
from scipy.optimize import linear_sum_assignment


# =========================
# 參數設定（四相機 + 可選配對）
# =========================
CALIB_ROOT   = "colcon_ws/src/4camera"
MODEL_PATH   = "colcon_ws/src/human.pt"
REID_WEIGHTS = "colcon_ws/src/osnet_x0_25_msmt17.pt"

# ---- 四相機 device index（依你的 /dev/video* 調整）----
CAM1_IDX = 2
CAM2_IDX = 4
CAM3_IDX = 6
CAM4_IDX = 8

# ---- 要使用哪些配對（在這裡切換）----
USE_PAIR12 = True
USE_PAIR34 = False

PAIR12_FOLDER = "pair12"
PAIR34_FOLDER = "pair34"

LIVE_SIZE    = (1920, 1080)
TARGET_FPS   = 30.0
DETECT_IMGSZ = 384
USE_GPU_HALF = True
BATCH_YOLO   = True

MAX_PERSONS  = 9
COLORS3D     = [(0.0, 0.0, 0.0)] * MAX_PERSONS

# =========================
# 新版配對邏輯參數
# =========================
DETECT_BOX_CONF_LOW  = 0.4
EPI_DIST_THRESH      = 200.0
PNEC_SIGMA           = 5.0
W_PNEC               = 1.0
W_REID               = 0.5
W_PROJ               = 0.5
STICKY_BONUS         = 0.5
STICKY_IOU_THRESH    = 0.35

# 3D 追蹤參數
TRACK3D_MAX_AGE      = 30
TRACK3D_DIST_THRESH  = 1500.0
PREDICT_DRAW_MAX_AGE = 5

# 高度過濾
MAX_TOTAL_HEIGHT_MM = 2300.0
MIN_TOTAL_HEIGHT_MM = 1000.0
MAX_CENTER_Z_MM     = 800.0

# ---- 兩組 pair 的對齊（如果你有既有對齊偏移需求）----
APPLY_ALIGN_PAIR34 = True
ALIGN_34_XY = np.array([-1050.0, 500.0], dtype=np.float64)

APPLY_ALIGN_PAIR12 = False
ALIGN_12_XY = np.array([0.0, 0.0], dtype=np.float64)  # 若 pair12 也要對齊請填入

# ---- 同時開雙 pair 時的 3D det 融合（避免同一人雙重 det）----
FUSE_DET_DIST_MM = 600.0  # 3D center 距離小於此值視為同一人，擇優保留

# 人體骨架連線
SKELETON_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (5,11),(6,12)
]

# =========================
# PyBullet / 安全區域參數（保留不動）
# =========================
FLOOR_TEXTURE_PATH = "colcon_ws/src/images.png"
PLATFORM_SIZE_M    = 0.20
PLATFORM_HEIGHT_M  = 0.69
FLOOR_Z            = -0.69

AABB_MARGIN_M = 0.05
EMA_ALPHA     = 0.25
AABB_SCALE    = 1.5

SLOW_BOX_CENTER = [0.0, 0.0, 0.0]
SLOW_BOX_HALF_EXTENT_M = 2.0
SLOW_BOX_COLOR = [0, 0, 1]
SLOW_BOX_ALERT = [1.0, 0.5, 0.0]
SLOW_BOX_LINE_WIDTH = 4

OVERLAY_LINE_WIDTH = 15
BBOX_LINE_WIDTH    = 2

# —— BIAS（mm）
BIAS_XY_M = (+1, 0.)
BIAS_MM = np.array([BIAS_XY_M[0]*1000.0, BIAS_XY_M[1]*1000.0, 0.0], dtype=np.float64)

# 取代藍盒的自訂平面遮罩（XY 正方形）
USE_XY_SQUARE_MASK = True
XY_MASK_CENTER_M   = (0.0, 0.0)
XY_MASK_HALF_M     = 2.5
XY_MASK_Z_RANGE_M  = None

# 暫停區冷卻時間
RESUME_COOLDOWN_SEC = 0.5
DRAW_HOLD_FR = 10
MIN_VALID_BONES_TO_DRAW = 5
EDGE_MAX_MM = 1100.0


# =========================
# 幾何與校正工具
# =========================
_id_re = re.compile(r'(\d+)')

def extract_id(path):
    m = _id_re.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def index_by_id(paths):
    idx, dup = {}, {}
    for pth in paths:
        k = extract_id(pth)
        if k is None: continue
        if k in idx: dup.setdefault(k, []).append(pth)
        else: idx[k] = pth
    if dup: raise RuntimeError(f"發現重複ID: {sorted(dup.items())}")
    return idx

def get_homogeneous_transform(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R; T[:3,3] = t.reshape(3)
    return T

def inverse_T(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3,:3] = R.T; Ti[:3,3] = -R.T @ t
    return Ti

def reorthonormalize_transform(T):
    R, t = T[:3,:3], T[:3,3]
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    Tn = np.eye(4, dtype=np.float64)
    Tn[:3,:3] = Rn
    Tn[:3,3]  = t
    return Tn

def calibrate_camera_from_folder(folder, chessboard_size=(4,3), square_size=71.0):
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
    objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
    objp[:,:2] = square_size*np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

    objpoints, imgpoints = [], []
    images = sorted(glob.glob(os.path.join(folder, '*.png')))
    if not images: raise RuntimeError(f"找不到標定資料夾：{folder}")
    gray_sz = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None: continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(g, chessboard_size, None)
        if not ret: continue
        corners2 = cv2.cornerSubPix(g, corners, (11,11), (-1,-1), criteria)
        objpoints.append(objp.copy())
        imgpoints.append(corners2)
        gray_sz = g.shape[::-1]

    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_sz, None, None)
    if not ret: raise RuntimeError("相機標定失敗")

    R_obj_cam, _ = cv2.Rodrigues(rvecs[0])
    t_obj_cam = tvecs[0].reshape(3,1)
    R_chess_cam = R_obj_cam.T
    t_chess_cam = (-R_obj_cam.T @ t_obj_cam).reshape(3,)
    return {
        'K': K, 'D': D, 'image_size': gray_sz,
        'R_chess_cam': R_chess_cam, 't_chess_cam': t_chess_cam
    }

def undistort_points_xy(kp, K, D):
    pts = kp.reshape(-1,1,2).astype(np.float32)
    return cv2.undistortPoints(pts, K, D).reshape(-1,2)

def project_base_to_pixel(X_base, T_c_base, K):
    Xb = np.asarray(X_base, dtype=np.float64).reshape(3,)
    R = T_c_base[:3,:3]; t = T_c_base[:3,3]
    Xc = R @ Xb + t
    if Xc[2] <= 1e-3: return None
    x = Xc[0] / Xc[2]; y = Xc[1] / Xc[2]
    u = K[0,0]*x + K[0,2]
    v = K[1,1]*y + K[1,2]
    return np.array([u, v], dtype=np.float64)

def bbox_iou(box1, box2):
    if box1 is None or box2 is None: return 0.0
    b1 = np.asarray(box1, dtype=np.float64).reshape(-1)
    b2 = np.asarray(box2, dtype=np.float64).reshape(-1)
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter_w = max(0.0, x2 - x1); inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0: return 0.0
    area1 = max(0.0, (b1[2] - b1[0])) * max(0.0, (b1[3] - b1[1]))
    area2 = max(0.0, (b2[2] - b2[0])) * max(0.0, (b2[3] - b2[1]))
    union = area1 + area2 - inter + 1e-6
    return float(inter / union)

def _resize_keep_max(img, target):
    h,w = img.shape[:2]
    if max(h,w)==target: return img.copy(), 1.0, 1.0
    scale = target/float(max(h,w))
    new_w, new_h = int(round(w*scale)), int(round(h*scale))
    imr = cv2.resize(img, (new_w,new_h), interpolation=cv2.INTER_LINEAR)
    return imr, (w/float(new_w)), (h/float(new_h))

def dets_from_res_scaled(res, sx=1.0, sy=1.0):
    dets = []
    if (res is None) or (not hasattr(res, 'boxes')) or len(res.boxes) == 0:
        return dets
    boxes = res.boxes.xyxy.detach().cpu().numpy().copy()
    boxes[:, [0, 2]] *= sx; boxes[:, [1, 3]] *= sy
    has_kp = hasattr(res, "keypoints") and res.keypoints is not None
    kps = res.keypoints.xy.detach().cpu().numpy().copy() if has_kp else None
    if kps is not None:
        kps[:, :, 0] *= sx
        kps[:, :, 1] *= sy
    kpc = res.keypoints.conf.detach().cpu().numpy().copy() if has_kp else None
    bconfs = res.boxes.conf.detach().cpu().numpy()
    for i in range(len(boxes)):
        if bconfs[i] < DETECT_BOX_CONF_LOW:
            continue
        dets.append({
            "kp": kps[i] if kps is not None else None,
            "kp_conf": kpc[i] if kpc is not None else None,
            "box": boxes[i],
            "score": float(bconfs[i]),
        })
    return dets


# =========================
# ReID: OSNet
# =========================
reid_model = None
reid_device = None

def init_reid(device_torch: torch.device):
    global reid_model, reid_device
    reid_device = device_torch
    if torchreid is None:
        print("[WARN] 未安裝 torchreid，跳過 ReID 初始化")
        reid_model = None
        return
    try:
        print("[INFO] 初始化 ReID (osnet_x0_25_msmt17)...")
        reid_model = torchreid.models.build_model(
            name='osnet_x0_25', num_classes=1041, pretrained=False
        )

        if os.path.exists(REID_WEIGHTS):
            state = torch.load(REID_WEIGHTS, map_location='cpu')
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            new_state = {}
            for k, v in state.items():
                if k.startswith('module.'): k = k[len('module.'):]
                if k.startswith('classifier.'):  # drop classifier head
                    continue
                new_state[k] = v
            reid_model.load_state_dict(new_state, strict=False)

            reid_model.to(device_torch)
            reid_model.float()
            reid_model.eval()
            print("[INFO] ReID 權重載入完成")
        else:
            print(f"[WARN] 找不到 ReID 權重檔: {REID_WEIGHTS}，ReID 功能將受限")
            reid_model = None
    except Exception as e:
        print(f"[WARN] ReID 初始化失敗：{e}")
        reid_model = None

def compute_reid_for_dets(dets, frame):
    global reid_model, reid_device
    if reid_model is None or len(dets) == 0:
        return
    h, w = frame.shape[:2]
    patches = []
    owners = []

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for idx, d in enumerate(dets):
        box = d["box"]
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop = cv2.resize(crop, (128, 256))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = crop.astype(np.float32) / 255.0
        crop = (crop - mean) / std
        crop = np.transpose(crop, (2, 0, 1))
        patches.append(crop)
        owners.append(idx)

    if not patches:
        return

    tensor = torch.from_numpy(np.stack(patches, axis=0)).to(reid_device).float()
    with torch.no_grad():
        feats = reid_model(tensor)

    feats = feats.cpu().numpy()
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    feats = feats / norms

    for i, idx in enumerate(owners):
        dets[idx]["reid"] = feats[i]


# =========================
# Stereo Core: PNEC + ReID + 2-Stage
# =========================
def symmetrical_epipolar_distance(dL, dR, F):
    kp1 = dL.get("kp"); kp2 = dR.get("kp")
    kpc1 = dL.get("kp_conf"); kpc2 = dR.get("kp_conf")

    if kp1 is None or kp2 is None:
        bL = dL["box"]; bR = dR["box"]
        cL = 0.5 * (bL[:2] + bL[2:])
        cR = 0.5 * (bR[:2] + bR[2:])
        xL = np.array([cL[0], cL[1], 1.0])
        xR = np.array([cR[0], cR[1], 1.0])
        lR = F @ xL
        lL = F.T @ xR
        dR_epi = abs(xR @ lR) / (math.hypot(lR[0], lR[1]) + 1e-6)
        dL_epi = abs(xL @ lL) / (math.hypot(lL[0], lL[1]) + 1e-6)
        return 0.5 * (dR_epi + dL_epi)

    K_len = min(kp1.shape[0], kp2.shape[0])
    total, cnt = 0.0, 0
    for idx in range(K_len):
        if kpc1 is not None and kpc1[idx] <= 0: continue
        if kpc2 is not None and kpc2[idx] <= 0: continue
        xL = np.array([kp1[idx,0], kp1[idx,1], 1.0])
        xR = np.array([kp2[idx,0], kp2[idx,1], 1.0])
        lR = F @ xL
        lL = F.T @ xR
        dR_epi = abs(xR @ lR) / (math.hypot(lR[0], lR[1]) + 1e-6)
        dL_epi = abs(xL @ lL) / (math.hypot(lL[0], lL[1]) + 1e-6)
        total += (dR_epi + dL_epi)
        cnt += 1
    if cnt == 0:
        return None
    return total / cnt

def build_stereo_cost(
    dL, dR, F, img_size,
    track_projs=None, track_boxes=None,
    w_pnec=1.0, sigma_pnec=5.0, w_reid=1.0, w_proj=0.3,
    epi_dist_thresh=EPI_DIST_THRESH
):
    H, W = img_size[1], img_size[0]
    n1, n2 = len(dL), len(dR)
    if n1 == 0 or n2 == 0:
        return np.zeros((0, 0), dtype=np.float64)

    LARGE = 1e6
    C = np.zeros((n1, n2), dtype=np.float64)
    Dnorm = math.hypot(W, H) + 1e-6

    track_projs_items = []
    if track_projs:
        for tid, pr in track_projs.items():
            if "L" in pr and "R" in pr:
                track_projs_items.append((tid, np.asarray(pr["L"]), np.asarray(pr["R"])))

    sticky_items = []
    if track_boxes:
        for tid, br in track_boxes.items():
            if "L" in br and "R" in br:
                sticky_items.append((tid, np.asarray(br["L"]), np.asarray(br["R"])))

    for i in range(n1):
        bL = dL[i]["box"]; cL = 0.5 * (bL[:2] + bL[2:])
        for j in range(n2):
            bR = dR[j]["box"]; cR = 0.5 * (bR[:2] + bR[2:])

            Dsym = symmetrical_epipolar_distance(dL[i], dR[j], F)
            if Dsym is None:
                xL = np.array([cL[0], cL[1], 1.0]); xR = np.array([cR[0], cR[1], 1.0])
                lR = F @ xL; lL = F.T @ xR
                Dsym = 0.5 * (
                    abs(xR @ lR)/math.hypot(lR[0],lR[1]) +
                    abs(xL @ lL)/math.hypot(lL[0],lL[1])
                )

            if Dsym > epi_dist_thresh:
                C[i, j] = LARGE
                continue

            cost_epi = 1.0 - math.exp(-(Dsym**2) / (2.0 * sigma_pnec**2))

            cost_reid = 0.0
            if "reid" in dL[i] and "reid" in dR[j]:
                cost_reid = float(np.linalg.norm(dL[i]["reid"] - dR[j]["reid"]))

            cost_proj = 0.0
            if track_projs_items:
                d_proj_min = None
                for _, pL, pR in track_projs_items:
                    d_ = np.linalg.norm(cL - pL) + np.linalg.norm(cR - pR)
                    if (d_proj_min is None) or (d_ < d_proj_min):
                        d_proj_min = d_
                if d_proj_min is not None:
                    cost_proj = (d_proj_min / Dnorm)

            cost_sticky = 0.0
            if sticky_items:
                for _, bL_prev, bR_prev in sticky_items:
                    if bbox_iou(bL, bL_prev) >= STICKY_IOU_THRESH and bbox_iou(bR, bR_prev) >= STICKY_IOU_THRESH:
                        cost_sticky = -STICKY_BONUS
                        break

            C[i, j] = w_pnec * cost_epi + w_reid * cost_reid + w_proj * cost_proj + cost_sticky

    return C

def triangulate_skeleton_norm(kp1_norm, kp2_norm, P1n, P2n):
    pts1 = kp1_norm.T.astype(np.float32)
    pts2 = kp2_norm.T.astype(np.float32)
    ph = cv2.triangulatePoints(P1n, P2n, pts1, pts2)
    return (ph[:3] / (ph[3] + 1e-12)).T

def triangulate_center_simple(dL, dR, K1, D1, K2, D2, P1n, P2n):
    kp1, kp2 = dL.get("kp"), dR.get("kp")
    cf1, cf2 = dL.get("kp_conf"), dR.get("kp_conf")
    X_full = np.full((17,3), np.nan, dtype=np.float64)
    center = None

    if kp1 is not None and kp2 is not None:
        valid = np.isfinite(kp1[...,0]) & np.isfinite(kp2[...,0])
        if cf1 is not None and cf2 is not None:
            valid &= (cf1 > 0) & (cf2 > 0)
        if np.count_nonzero(valid) >= 2:
            kp1n_v = undistort_points_xy(kp1[valid], K1, D1)
            kp2n_v = undistort_points_xy(kp2[valid], K2, D2)
            Xw = triangulate_skeleton_norm(kp1n_v, kp2n_v, P1n, P2n)
            vi = np.where(valid)[0]
            for ii,k in enumerate(vi):
                if k < 17:
                    X_full[k] = Xw[ii]

            def _ok(idx): return (idx is not None) and (0<=idx<17) and np.all(np.isfinite(X_full[idx]))
            if _ok(11) and _ok(12): center = 0.5*(X_full[11]+X_full[12])
            elif _ok(5) and _ok(6): center = 0.5*(X_full[5]+X_full[6])
            elif _ok(0): center = X_full[0]

    if center is None:
        cL = 0.5*(dL["box"][:2] + dL["box"][2:])
        cR = 0.5*(dR["box"][:2] + dR["box"][2:])
        kp1n = undistort_points_xy(cL.reshape(1,2), K1, D1)
        kp2n = undistort_points_xy(cR.reshape(1,2), K2, D2)
        Xc = triangulate_skeleton_norm(kp1n, kp2n, P1n, P2n).reshape(3)
        if np.all(np.isfinite(Xc)):
            center = Xc

    if center is None or not np.all(np.isfinite(center)):
        return None, X_full
    return center, X_full

def estimate_total_height_mm(X_full_mm):
    if X_full_mm is None or X_full_mm.shape[0] == 0:
        return None
    valid = np.isfinite(X_full_mm).all(axis=1)
    if np.count_nonzero(valid) < 2:
        return None
    z_vals = X_full_mm[valid, 2]
    return float(z_vals.max() - z_vals.min())

def estimate_center_height_mm_from_extremes(X_full_mm):
    if X_full_mm is None or X_full_mm.shape[0] == 0:
        return None
    valid = np.isfinite(X_full_mm).all(axis=1)
    z_vals = X_full_mm[valid, 2]
    if z_vals.size == 0:
        return None
    z_sorted = np.sort(z_vals)
    if z_sorted.size >= 4:
        lowest2  = z_sorted[:2]
        highest2 = z_sorted[-2:]
        return float((lowest2.sum() + highest2.sum()) / 4.0)
    else:
        return float(0.5 * (z_sorted.min() + z_sorted.max()))

def _apply_align_xy(center_mm: np.ndarray, X_full_mm: np.ndarray, align_xy: np.ndarray):
    if center_mm is not None and np.all(np.isfinite(center_mm)):
        center_mm = center_mm.copy()
        center_mm[0] += float(align_xy[0])
        center_mm[1] += float(align_xy[1])
    if X_full_mm is not None and X_full_mm.size > 0:
        X_full_mm = X_full_mm.copy()
        good = np.isfinite(X_full_mm).all(axis=1)
        X_full_mm[good, 0] += float(align_xy[0])
        X_full_mm[good, 1] += float(align_xy[1])
    return center_mm, X_full_mm

def pair_passes_3d_constraints(iL, iR, dL, dR, cfg):
    c3d, X = triangulate_center_simple(
        dL[iL], dR[iR],
        cfg.camL['K'], cfg.camL['D'], cfg.camR['K'], cfg.camR['D'],
        cfg.P_L, cfg.P_R
    )
    if c3d is None:
        return False
    if cfg.apply_align:
        c3d, X = _apply_align_xy(c3d, X, cfg.align_xy)

    height_mm = estimate_total_height_mm(X)
    if height_mm is not None and ((height_mm < MIN_TOTAL_HEIGHT_MM) or (height_mm > MAX_TOTAL_HEIGHT_MM)):
        return False

    center_z_new = estimate_center_height_mm_from_extremes(X)
    if center_z_new is not None and center_z_new > MAX_CENTER_Z_MM:
        return False

    return True

def tri_from_pairs(pairs, dL, dR, cfg):
    det3d_list = []
    for (iL, iR, cost) in pairs:
        c3d, X = triangulate_center_simple(
            dL[iL], dR[iR],
            cfg.camL['K'], cfg.camL['D'], cfg.camR['K'], cfg.camR['D'],
            cfg.P_L, cfg.P_R
        )
        if c3d is None:
            continue

        if cfg.apply_align:
            c3d, X = _apply_align_xy(c3d, X, cfg.align_xy)

        height_mm = estimate_total_height_mm(X)
        if height_mm is not None:
            if (height_mm < MIN_TOTAL_HEIGHT_MM) or (height_mm > MAX_TOTAL_HEIGHT_MM):
                continue

        center_z_new = estimate_center_height_mm_from_extremes(X)
        if center_z_new is not None:
            if center_z_new > MAX_CENTER_Z_MM:
                continue
            c3d[2] = center_z_new

        boxes_by_cam = {
            cfg.cam_name_L: dL[iL]["box"].copy(),
            cfg.cam_name_R: dR[iR]["box"].copy(),
        }

        det3d_list.append({
            "center": np.asarray(c3d, dtype=np.float64),
            "X_full": np.asarray(X, dtype=np.float64),
            "boxes_by_cam": boxes_by_cam,
            "src_pair": cfg.pair_name,
        })
    return det3d_list

def stereo_triangulate_two_stage(dL, dR, cfg, track_projs=None, track_boxes=None):
    n1, n2 = len(dL), len(dR)
    if n1 == 0 or n2 == 0:
        return []

    C = build_stereo_cost(
        dL, dR, cfg.F, cfg.camL['image_size'],
        track_projs=track_projs, track_boxes=track_boxes,
        w_pnec=W_PNEC, sigma_pnec=PNEC_SIGMA, w_reid=W_REID, w_proj=W_PROJ,
        epi_dist_thresh=EPI_DIST_THRESH
    )

    LARGE = 1e6
    rows, cols = linear_sum_assignment(C)
    pairs0 = [(int(r), int(c), float(C[r, c])) for r, c in zip(rows, cols)]
    pairs0.sort(key=lambda x: x[2])

    valid_pairs, invalid_pairs = [], []
    used_rows, used_cols = set(), set()

    # Stage 1
    for (iL, iR, cost) in pairs0:
        if pair_passes_3d_constraints(iL, iR, dL, dR, cfg):
            valid_pairs.append((iL, iR, cost))
            used_rows.add(iL); used_cols.add(iR)
        else:
            invalid_pairs.append((iL, iR))

    for (iL, iR) in invalid_pairs:
        C[iL, iR] = LARGE

    # Stage 2 rematch on remaining
    rows_free = sorted(set(range(n1)) - used_rows)
    cols_free = sorted(set(range(n2)) - used_cols)

    if rows_free and cols_free:
        C_sub = C[np.ix_(rows_free, cols_free)]
        r_sub, c_sub = linear_sum_assignment(C_sub)
        for rr, cc in zip(r_sub, c_sub):
            iL = rows_free[rr]
            iR = cols_free[cc]
            cost = float(C[iL, iR])
            if pair_passes_3d_constraints(iL, iR, dL, dR, cfg):
                valid_pairs.append((iL, iR, cost))

    if not valid_pairs:
        return []
    return tri_from_pairs(valid_pairs, dL, dR, cfg)

def fuse_det3d_list(det_list, dist_mm=FUSE_DET_DIST_MM):

    if not det_list:
        return []

    kept = []
    for d in det_list:
        c = d["center"]
        if c is None or not np.all(np.isfinite(c)):
            continue

        best_k = -1
        best_idx = None
        for i, k in enumerate(kept):
            ck = k["center"]
            if np.linalg.norm(c - ck) <= dist_mm:
                best_idx = i
                break

        def score_det(dd):
            X = dd.get("X_full")
            if X is None or X.size == 0:
                return 0
            return int(np.isfinite(X).all(axis=1).sum())

        if best_idx is None:
            kept.append(d)
        else:
            if score_det(d) > score_det(kept[best_idx]):
                kept[best_idx] = d

    return kept


# =========================
# Kalman Filter & Tracker 3D
# =========================
class Kalman3D:
    def __init__(self, center3d):
        cx, cy, cz = center3d
        self.x = np.array([[cx],[cy],[cz],[0.0],[0.0],[0.0]], dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 100.0
        self.Q = np.eye(6, dtype=np.float64) * 1.0
        self.R = np.eye(3, dtype=np.float64) * 10.0

    def predict(self, dt=1.0):
        F = np.array([
            [1,0,0,dt,0,0],
            [0,1,0,0,dt,0],
            [0,0,1,0,0,dt],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1],
        ], dtype=np.float64)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        H = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,0,0,0],
        ], dtype=np.float64)
        z = np.asarray(z, dtype=np.float64).reshape(3,1)
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P

    def get_pos(self):
        return self.x[0:3,0].copy()

class Track3D:
    def __init__(self, track_id, det3d):
        self.id = track_id
        self.kf = Kalman3D(det3d["center"])
        self.center = det3d["center"].copy()
        self.X_full_mm = det3d["X_full"].copy()
        self.boxes_by_cam = dict(det3d.get("boxes_by_cam", {}))
        self.age = 1
        self.time_since_update = 0

    def predict(self, dt=1.0):
        old_center = self.center.copy()
        self.kf.predict(dt)
        self.center = self.kf.get_pos()
        delta = self.center - old_center
        if self.X_full_mm is not None and np.isfinite(self.X_full_mm).any():
            valid = np.isfinite(self.X_full_mm).all(axis=1)
            self.X_full_mm[valid] += delta
        self.age += 1
        self.time_since_update += 1

    def update(self, det3d):
        self.kf.update(det3d["center"])
        self.center = self.kf.get_pos()
        self.X_full_mm = det3d["X_full"].copy()
        self.boxes_by_cam = dict(det3d.get("boxes_by_cam", {}))
        self.time_since_update = 0

    def get_pred_center(self):
        return self.kf.get_pos()

class MultiObjectTracker3D:
    def __init__(self, max_age=TRACK3D_MAX_AGE, dist_thresh=TRACK3D_DIST_THRESH):
        self.max_age = max_age
        self.dist_thresh = dist_thresh
        self.tracks = []
        self._next_id = 1

    def predict(self, dt=1.0):
        for t in self.tracks:
            t.predict(dt)

    def update(self, det3d_list):
        if len(self.tracks) == 0:
            for d in det3d_list:
                self.tracks.append(Track3D(self._next_id, d))
                self._next_id += 1
        else:
            nT = len(self.tracks)
            nD = len(det3d_list)
            if nD > 0:
                C = np.zeros((nT, nD), dtype=np.float64)
                for i, tr in enumerate(self.tracks):
                    p = tr.get_pred_center()
                    for j, d in enumerate(det3d_list):
                        C[i,j] = np.linalg.norm(p - d["center"])

                rows, cols = linear_sum_assignment(C)
                matched_d = set()
                for r, c in zip(rows, cols):
                    if C[r,c] <= self.dist_thresh:
                        self.tracks[r].update(det3d_list[c])
                        matched_d.add(c)

                for j in range(nD):
                    if j not in matched_d:
                        self.tracks.append(Track3D(self._next_id, det3d_list[j]))
                        self._next_id += 1

            self.tracks = [tr for tr in self.tracks if tr.time_since_update <= self.max_age]

        active_tracks = [t for t in self.tracks if t.time_since_update <= PREDICT_DRAW_MAX_AGE]
        return active_tracks


# =========================
# PyBullet 模擬器（未更動）
# =========================
class TMSimulator:
    def __init__(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0,0,-9.81)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        self.joint_indices=[1,2,3,4,5,6]
        self._add_floor_and_platform()

        self.tm5_id = p.loadURDF(
            "colcon_ws/src/tmr_ros2/tm_description/urdf/tm5-900.urdf",
            basePosition=[0,0,0], useFixedBase=True
        )
        nj = p.getNumJoints(self.tm5_id)
        p.changeVisualShape(self.tm5_id, -1, rgbaColor=[1,1,1,1])
        for link_idx in range(nj):
            p.changeVisualShape(self.tm5_id, link_idx, rgbaColor=[1,1,1,1])

        self.last_green_exit = None
        self.last_slow_exit = None
        self.hold_max = DRAW_HOLD_FR
        self.hold_age = [0] * MAX_PERSONS
        self.last_pts3d_m = [None] * MAX_PERSONS
        self.last_valid_mask = [None] * MAX_PERSONS

        self._bbox_lock = threading.Lock()
        self.current_bounds = [-1.1, 1.1, -1.1, 1.1, FLOOR_Z, 1.3]
        self._ema_c = None
        self._ema_r = None
        self.ema_alpha = EMA_ALPHA
        self.aabb_scale = AABB_SCALE
        self.bbox_ids=None
        self.overlay_ids=None
        self.set_bbox_bounds(*self.current_bounds, color=[0,1,0], line_width=BBOX_LINE_WIDTH)

        self.slow_box_ids=None
        self.slow_bounds=None
        cx,cy,cz = SLOW_BOX_CENTER
        r = float(SLOW_BOX_HALF_EXTENT_M)
        self.set_slow_bbox_bounds(cx-r, cx+r, cy-r, cy+r, cz-r, cz+r,
                                  color=SLOW_BOX_COLOR, line_width=SLOW_BOX_LINE_WIDTH)

        self.skel_line_ids = []
        for _ in range(MAX_PERSONS):
            line_set=[]
            for _ in SKELETON_EDGES:
                lid = p.addUserDebugLine([0,0,0],[0,0,0],[0,0,0],6,0)
                line_set.append(lid)
            self.skel_line_ids.append(line_set)
        self.prev_valid = [[False]*len(SKELETON_EDGES) for _ in range(MAX_PERSONS)]

    def _add_floor_and_platform(self):
        plane_id = p.loadURDF("plane.urdf")
        try:
            if os.path.isfile(FLOOR_TEXTURE_PATH):
                tex = p.loadTexture(FLOOR_TEXTURE_PATH)
                p.changeVisualShape(plane_id, -1, textureUniqueId=tex)
        except Exception:
            pass
        p.resetBasePositionAndOrientation(plane_id,[0,0,FLOOR_Z],[0,0,0,1])

        size=float(PLATFORM_SIZE_M)
        height=float(PLATFORM_HEIGHT_M)
        half=[size/2,size/2,height/2]
        col=p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis=p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.6,0.6,0.6,1])
        base_pos=[0,0,FLOOR_Z+height/2]
        p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col,
                          baseVisualShapeIndex=vis, basePosition=base_pos)

    def _edges_from_bounds(self, xm,xM,ym,yM,zm,zM):
        return [
            ([xm,ym,zm],[xM,ym,zm]), ([xM,ym,zm],[xM,yM,zm]),
            ([xM,yM,zm],[xm,yM,zm]), ([xm,ym,zm],[xm,ym,zm]),
            ([xm,ym,zM],[xM,ym,zM]), ([xM,ym,zM],[xM,yM,zM]),
            ([xM,yM,zM],[xm,yM,zM]), ([xm,yM,zM],[xm,ym,zM]),
            ([xm,ym,zm],[xm,ym,zM]), ([xM,ym,zm],[xM,ym,zM]),
            ([xM,yM,zm],[xM,yM,zM]), ([xm,yM,zm],[xm,yM,zM])
        ]

    def set_bbox_bounds(self, xm,xM,ym,yM,zm,zM, color=[0,1,0], line_width=BBOX_LINE_WIDTH):
        edges=self._edges_from_bounds(xm,xM,ym,yM,zm,zM)
        with self._bbox_lock:
            self.current_bounds=[xm,xM,ym,yM,zm,zM]
            if self.bbox_ids is None:
                self.bbox_ids=[]
                for s,e in edges:
                    lid=p.addUserDebugLine(s,e,color,line_width,0)
                    self.bbox_ids.append(lid)
            else:
                for idx,(s,e) in enumerate(edges):
                    p.addUserDebugLine(s,e,color,line_width,0,replaceItemUniqueId=self.bbox_ids[idx])

    def update_bounding_box_color(self, color, line_width):
        with self._bbox_lock:
            xm,xM,ym,yM,zm,zM=self.current_bounds
        edges=self._edges_from_bounds(xm,xM,ym,yM,zm,zM)
        for idx,(s,e) in enumerate(edges):
            p.addUserDebugLine(s,e,color,line_width,0,replaceItemUniqueId=self.bbox_ids[idx])

    def show_overlay_from_current(self, color=[1,0,0], line_width=OVERLAY_LINE_WIDTH):
        with self._bbox_lock:
            xm,xM,ym,yM,zm,zM = self.current_bounds
            edges=self._edges_from_bounds(xm,xM,ym,yM,zm,zM)
            if self.overlay_ids is None:
                self.overlay_ids=[]
                for s,e in edges:
                    lid=p.addUserDebugLine(s,e,color,line_width,0)
                    self.overlay_ids.append(lid)
            else:
                for idx,(s,e) in enumerate(edges):
                    p.addUserDebugLine(s,e,color,line_width,0,replaceItemUniqueId=self.overlay_ids[idx])

    def hide_overlay(self):
        if self.overlay_ids is not None:
            for lid in self.overlay_ids:
                p.addUserDebugLine([0,0,0],[0,0,0],[0,0,0],1,0,replaceItemUniqueId=lid)
            self.overlay_ids=None

    def set_slow_bbox_bounds(self, xm,xM,ym,yM,zm,zM, color=SLOW_BOX_COLOR, line_width=SLOW_BOX_LINE_WIDTH):
        edges=self._edges_from_bounds(xm,xM,ym,yM,zm,zM)
        with self._bbox_lock:
            self.slow_bounds=[xm,xM,ym,yM,zm,zM]
            if self.slow_box_ids is None:
                self.slow_box_ids=[]
                for s,e in edges:
                    lid=p.addUserDebugLine(s,e,color,line_width,0)
                    self.slow_box_ids.append(lid)
            else:
                for idx,(s,e) in enumerate(edges):
                    p.addUserDebugLine(s,e,color,line_width,0,replaceItemUniqueId=self.slow_box_ids[idx])

    def update_slow_box_color(self, color=SLOW_BOX_COLOR, line_width=SLOW_BOX_LINE_WIDTH):
        with self._bbox_lock:
            if self.slow_bounds is None or self.slow_box_ids is None:
                return
            xm,xM,ym,yM,zm,zM = self.slow_bounds
        edges=self._edges_from_bounds(xm,xM,ym,yM,zm,zM)
        for idx,(s,e) in enumerate(edges):
            p.addUserDebugLine(s,e,color,line_width,0,replaceItemUniqueId=self.slow_box_ids[idx])

    def get_slow_bounds(self):
        with self._bbox_lock:
            return tuple(self.slow_bounds) if self.slow_bounds else None

    def update_joint_angles(self, joint_angles):
        for i,angle in enumerate(joint_angles):
            p.resetJointState(self.tm5_id, self.joint_indices[i], angle)

        num_links = p.getNumJoints(self.tm5_id)
        mn, mx = np.array(p.getAABB(self.tm5_id,-1)[0]), np.array(p.getAABB(self.tm5_id,-1)[1])
        for link_idx in range(num_links):
            aabb = p.getAABB(self.tm5_id, link_idx)
            mn = np.minimum(mn, aabb[0])
            mx = np.maximum(mx, aabb[1])

        mn -= AABB_MARGIN_M
        mx += AABB_MARGIN_M

        c_raw=(mn+mx)/2.0
        r_raw=np.max((mx-mn)/2.0)*float(self.aabb_scale)

        if self._ema_c is None:
            self._ema_c, self._ema_r = c_raw, r_raw
        else:
            a = self.ema_alpha
            self._ema_c = (1-a)*self._ema_c + a*c_raw
            self._ema_r = (1-a)*self._ema_r + a*r_raw

        mn2=self._ema_c - self._ema_r
        mx2=self._ema_c + self._ema_r
        xm,ym,zm = mn2.tolist()
        xM,yM,zM = mx2.tolist()
        self.set_bbox_bounds(xm,xM,ym,yM,zm,zM)

    def draw_skeleton_slot(self, slot_idx, pts3d_m, valid_mask, color_rgb):
        if not (0 <= slot_idx < len(self.skel_line_ids)):
            return
        line_set = self.skel_line_ids[slot_idx]
        for k,(i,j) in enumerate(SKELETON_EDGES):
            is_valid = (i < len(valid_mask) and j < len(valid_mask) and valid_mask[i] and valid_mask[j])
            if is_valid:
                p1 = pts3d_m[i].tolist()
                p2 = pts3d_m[j].tolist()
                p.addUserDebugLine(p1, p2, lineColorRGB=color_rgb, lineWidth=4, lifeTime=0,
                                   replaceItemUniqueId=line_set[k])
            else:
                p.addUserDebugLine([0,0,0],[0,0,0], lineColorRGB=color_rgb, lineWidth=1, lifeTime=0,
                                   replaceItemUniqueId=line_set[k])
            self.prev_valid[slot_idx][k] = is_valid

    def draw_skeleton_slot_hold(self, slot_idx, pts3d_m, valid_mask, color_rgb):
        if not (0 <= slot_idx < len(self.skel_line_ids)):
            return

        use_pts = pts3d_m
        use_mask = valid_mask
        has_new = (pts3d_m is not None) and (valid_mask is not None) and np.any(valid_mask)

        if has_new:
            self.last_pts3d_m[slot_idx] = np.array(pts3d_m, dtype=np.float64, copy=True)
            self.last_valid_mask[slot_idx] = np.array(valid_mask, dtype=bool, copy=True)
            self.hold_age[slot_idx] = 0
        else:
            if (self.last_pts3d_m[slot_idx] is not None) and (self.last_valid_mask[slot_idx] is not None) and (self.hold_age[slot_idx] < self.hold_max):
                use_pts = self.last_pts3d_m[slot_idx]
                use_mask = self.last_valid_mask[slot_idx]
                self.hold_age[slot_idx] += 1
            else:
                self.clear_skeleton_slot(slot_idx)
                self.last_pts3d_m[slot_idx] = None
                self.last_valid_mask[slot_idx] = None
                self.hold_age[slot_idx] = 0
                return

        line_set = self.skel_line_ids[slot_idx]
        for k,(i,j) in enumerate(SKELETON_EDGES):
            ok = (use_mask is not None and i < len(use_mask) and j < len(use_mask) and use_mask[i] and use_mask[j])
            if ok:
                p1 = use_pts[i].tolist()
                p2 = use_pts[j].tolist()
                p.addUserDebugLine(p1, p2, lineColorRGB=color_rgb, lineWidth=4, lifeTime=0,
                                   replaceItemUniqueId=line_set[k])
            else:
                p.addUserDebugLine([0,0,0],[0,0,0], lineColorRGB=color_rgb, lineWidth=1, lifeTime=0,
                                   replaceItemUniqueId=line_set[k])
            self.prev_valid[slot_idx][k] = ok

    def clear_skeleton_slot(self, slot_idx):
        if not (0 <= slot_idx < len(self.skel_line_ids)):
            return
        for lid in self.skel_line_ids[slot_idx]:
            p.addUserDebugLine([0,0,0],[0,0,0],[0,0,0],1,0,replaceItemUniqueId=lid)
        self.prev_valid[slot_idx] = [False]*len(SKELETON_EDGES)

    def step(self):
        p.stepSimulation()
        time.sleep(1.0/240)


# =========================
# Pair Config
# =========================
@dataclass
class PairConfig:
    pair_name: str
    cam_name_L: str
    cam_name_R: str
    camL: dict
    camR: dict
    P_L: np.ndarray  # 3x4 world(base)->cam
    P_R: np.ndarray
    T_L_base: np.ndarray  # 4x4 world(base)->cam
    T_R_base: np.ndarray
    F: np.ndarray
    apply_align: bool
    align_xy: np.ndarray


# =========================
# ROS2 節點（四相機、可選 pair）
# =========================
class MultiHumanSafetyNode(Node):
    def __init__(self, sim, pair_cfgs, yolo_pack, caps):
        super().__init__('four_cam_human_safety_new_logic')
        self.sim = sim
        self.pair_cfgs = pair_cfgs   # dict: name -> PairConfig
        self.caps = caps             # dict: cam_name -> cv2.VideoCapture

        (self.model, self.yolo_device_arg, self.torch_device) = yolo_pack

        # ROS2 services
        self.send_cli = self.create_client(SendScript, '/send_script')
        while not self.send_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning("Waiting for /send_script...")
        self.event_cli = self.create_client(SetEvent, '/set_event')
        while not self.event_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning("Waiting for /set_event...")
        self.create_subscription(FeedbackState, '/feedback_states', self.feedback_cb, 10)

        # 運動腳本
        self.tcp_points_fast = [
            'PTP("JPP",-70,45,60,-10,90,0,100,100,100,false)',
            'PTP("JPP",0,45,60,-10,90,0,100,100,100,false)',
            'PTP("JPP",-70,45,60,-10,90,0,100,100,100,false)',
            'PTP("JPP",0,45,60,-10,0,100,100,100,false)'
        ]
        self.tcp_points_slow = [
            'PTP("JPP",-70,45,60,-10,90,0,40,100,100,false)',
            'PTP("JPP",0,45,60,-10,90,0,40,100,100,false)',
            'PTP("JPP",-70,45,60,-10,90,0,40,100,100,false)',
            'PTP("JPP",0,45,60,-10,90,0,40,100,100,false)'
        ]
        self.tcp_points = self.tcp_points_fast
        self.paused = False
        self.slow_mode = False
        self.lock = threading.Lock()

        # 3D 追蹤器
        self.tracker3d = MultiObjectTracker3D(max_age=TRACK3D_MAX_AGE, dist_thresh=TRACK3D_DIST_THRESH)

        self._shutdown = False

        sb = self.sim.get_slow_bounds()
        if sb:
            xm,xM,ym,yM,zm,zM = sb
            self.get_logger().info(
                f"固定減速盒：中心(0,0,0)，半徑={SLOW_BOX_HALF_EXTENT_M} m，"
                f"x∈[{xm:.2f},{xM:.2f}], y∈[{ym:.2f},{yM:.2f}], z∈[{zm:.2f},{zM:.2f}]"
            )

        threading.Thread(target=self.detect_loop, daemon=True).start()
        threading.Thread(target=self.run_loop, daemon=True).start()

    def _read_frames(self):
        frames = {}
        for cam_name, cap in self.caps.items():
            ok, f = cap.read()
            if not ok:
                frames[cam_name] = None
                continue
            if (f.shape[1], f.shape[0]) != LIVE_SIZE:
                f = cv2.resize(f, LIVE_SIZE)
            frames[cam_name] = f
        return frames

    def run_yolo_multi(self, frames_dict):
        cam_names = [k for k,v in frames_dict.items() if v is not None]
        if not cam_names:
            return {}

        ims_resized = []
        scales = []
        for cn in cam_names:
            imr, sx, sy = _resize_keep_max(frames_dict[cn], DETECT_IMGSZ)
            ims_resized.append(imr)
            scales.append((sx, sy))

        with torch.inference_mode():
            if BATCH_YOLO and len(ims_resized) >= 2:
                results = self.model.predict(
                    ims_resized, imgsz=DETECT_IMGSZ, device=self.yolo_device_arg,
                    verbose=False, conf=DETECT_BOX_CONF_LOW
                )
            else:
                results = []
                for imr in ims_resized:
                    results.append(self.model.predict(
                        imr, imgsz=DETECT_IMGSZ, device=self.yolo_device_arg,
                        verbose=False, conf=DETECT_BOX_CONF_LOW
                    )[0])

        dets_by_cam = {}
        for i, cn in enumerate(cam_names):
            sx, sy = scales[i]
            dets_by_cam[cn] = dets_from_res_scaled(results[i], sx=sx, sy=sy)

        return dets_by_cam

    def detect_loop(self):
        dt = 1.0 / TARGET_FPS

        while rclpy.ok() and not self._shutdown:
            # 1) tracker predict
            self.tracker3d.predict(dt)

            # 2) read frames
            frames = self._read_frames()
            if any(frames[k] is None for k in frames):
                time.sleep(0.001)
                continue

            # 3) YOLO + ReID (per cam)
            dets_by_cam = self.run_yolo_multi(frames)
            for cn, dets in dets_by_cam.items():
                compute_reid_for_dets(dets, frames[cn])

            # 4) 對每個啟用 pair 做 stereo + 3D constraints
            all_det3d = []

            for pair_name, cfg in self.pair_cfgs.items():
                dL = dets_by_cam.get(cfg.cam_name_L, [])
                dR = dets_by_cam.get(cfg.cam_name_R, [])

                # track projections / sticky boxes（針對此 pair）
                track_projs = {}
                track_boxes = {}
                for tr in self.tracker3d.tracks:
                    c_pred = tr.get_pred_center()
                    pL = project_base_to_pixel(c_pred, cfg.T_L_base, cfg.camL['K'])
                    pR = project_base_to_pixel(c_pred, cfg.T_R_base, cfg.camR['K'])
                    if pL is not None and pR is not None:
                        track_projs[tr.id] = {"L": pL, "R": pR}

                    bL = tr.boxes_by_cam.get(cfg.cam_name_L, None)
                    bR = tr.boxes_by_cam.get(cfg.cam_name_R, None)
                    if bL is not None and bR is not None:
                        track_boxes[tr.id] = {"L": bL, "R": bR}

                det3d_list = stereo_triangulate_two_stage(
                    dL, dR, cfg, track_projs=track_projs, track_boxes=track_boxes
                )
                all_det3d.extend(det3d_list)

            # 5) 若雙 pair 同開：融合 det
            if len(self.pair_cfgs) >= 2:
                all_det3d = fuse_det3d_list(all_det3d, dist_mm=FUSE_DET_DIST_MM)

            # 6) tracker update
            active_tracks = self.tracker3d.update(all_det3d)

            # 7) 轉成 skeleton list
            skeletons_to_check = [tr.X_full_mm for tr in active_tracks]
            if len(skeletons_to_check) > MAX_PERSONS:
                skeletons_to_check = skeletons_to_check[:MAX_PERSONS]

            # 8) render + zone check
            self._render_and_zone_check_simple(skeletons_to_check)

        for cap in self.caps.values():
            try:
                cap.release()
            except Exception:
                pass

    def _render_and_zone_check_simple(self, skeletons_mm_list):
        xg_min, xg_max, yg_min, yg_max, zg_min, zg_max = self.sim.current_bounds
        sb = self.sim.get_slow_bounds()
        if sb:
            xb_min, xb_max, yb_min, yb_max, zb_min, zb_max = sb
        else:
            xb_min = xb_max = yb_min = yb_max = zb_min = zb_max = 0.0

        any_in_green = False
        any_in_blue  = False

        for slot in range(MAX_PERSONS):
            color3d = COLORS3D[slot % len(COLORS3D)]
            X_full = skeletons_mm_list[slot] if slot < len(skeletons_mm_list) else None

            if X_full is None:
                self.sim.draw_skeleton_slot_hold(slot, None, None, list(color3d))
                continue

            X_draw_mm = X_full.copy()
            for i in range(17):
                if np.all(np.isfinite(X_draw_mm[i])):
                    X_draw_mm[i] = X_draw_mm[i] + BIAS_MM
            pts3d_m = (X_draw_mm / 1000.0)

            valid_mask = np.isfinite(pts3d_m).all(axis=1)

            if USE_XY_SQUARE_MASK:
                cx, cy = XY_MASK_CENTER_M
                hx = float(XY_MASK_HALF_M)
                hy = float(XY_MASK_HALF_M)
                inside_xy = np.zeros(17, dtype=bool)
                for i in range(17):
                    if valid_mask[i]:
                        x, y, z = pts3d_m[i]
                        ok_xy = (abs(x - cx) <= hx) and (abs(y - cy) <= hy)
                        if XY_MASK_Z_RANGE_M is None:
                            ok_z = True
                        else:
                            zmin, zmax = XY_MASK_Z_RANGE_M
                            ok_z = (zmin <= z <= zmax)
                        inside_xy[i] = (ok_xy and ok_z)
                    else:
                        inside_xy[i] = False

                for i in range(17):
                    if not inside_xy[i]:
                        pts3d_m[i] = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
                valid_mask = np.isfinite(pts3d_m).all(axis=1)

            xs, ys, zs = pts3d_m[:, 0], pts3d_m[:, 1], pts3d_m[:, 2]
            for (a, b) in SKELETON_EDGES:
                if a < 17 and b < 17 and np.isfinite(xs[a]) and np.isfinite(xs[b]):
                    seg = np.array([xs[a] - xs[b], ys[a] - ys[b], zs[a] - zs[b]], dtype=np.float64)
                    seg_len_mm = float(np.linalg.norm(seg)) * 1000.0
                    if not (np.isfinite(seg_len_mm) and seg_len_mm <= EDGE_MAX_MM):
                        pts3d_m[a] = np.array([np.nan, np.nan, np.nan])
                        pts3d_m[b] = np.array([np.nan, np.nan, np.nan])

            valid_mask = np.isfinite(pts3d_m).all(axis=1)
            nvalid_edges = sum(
                1 for (a, b) in SKELETON_EDGES
                if a < 17 and b < 17 and valid_mask[a] and valid_mask[b]
            )

            if nvalid_edges >= MIN_VALID_BONES_TO_DRAW:
                self.sim.draw_skeleton_slot(slot, pts3d_m, valid_mask, list(color3d))
            else:
                self.sim.draw_skeleton_slot_hold(slot, None, None, list(color3d))

            if np.any(valid_mask):
                if not any_in_green:
                    for (x, y, z) in pts3d_m[valid_mask]:
                        if xg_min <= x <= xg_max and yg_min <= y <= yg_max and zg_min <= z <= zg_max:
                            any_in_green = True
                            break

                if sb and not any_in_blue:
                    for (x, y, z) in pts3d_m[valid_mask]:
                        if xb_min <= x <= xb_max and yb_min <= y <= yb_max and zb_min <= z <= zb_max:
                            any_in_blue = True
                            break

        self._apply_zone_logic(any_in_green, any_in_blue)

    def _apply_zone_logic(self, in_green, in_blue):
        self.sim.update_bounding_box_color(color=[0,1,0], line_width=BBOX_LINE_WIDTH)
        now = time.time()

        with self.lock:
            if in_green:
                self.last_green_exit = None
                if not self.paused:
                    if self.event_cli.service_is_ready():
                        req = SetEvent.Request()
                        req.func = SetEvent.Request.PAUSE
                        req.arg0 = 0; req.arg1 = 0
                        self.event_cli.call_async(req)
                    self.get_logger().info("🚷 有人進入綠盒 → 暫停 & 顯示紅疊加")
                    self.paused = True
                    self.sim.show_overlay_from_current(color=[1,0,0], line_width=OVERLAY_LINE_WIDTH)
            else:
                if self.paused:
                    if self.last_green_exit is None:
                        self.last_green_exit = now
                        self.get_logger().info(f"⏳ 綠盒清空，開始冷卻 {RESUME_COOLDOWN_SEC:.1f}s 才恢復")
                    else:
                        elapsed = now - self.last_green_exit
                        if elapsed >= RESUME_COOLDOWN_SEC:
                            if self.event_cli.service_is_ready():
                                req = SetEvent.Request()
                                req.func = SetEvent.Request.RESUME
                                req.arg0 = 0; req.arg1 = 0
                                self.event_cli.call_async(req)
                            self.get_logger().info("✅ 冷卻完成 → 恢復 & 隱藏紅疊加")
                            self.paused = False
                            self.sim.hide_overlay()
                            self.last_green_exit = None

            if (in_blue and not self.slow_mode and not self.paused):
                self.slow_mode = True
                self.tcp_points = self.tcp_points_slow
                self.sim.update_slow_box_color(color=SLOW_BOX_ALERT, line_width=SLOW_BOX_LINE_WIDTH)
                self.get_logger().info("⚠️ 進入減速區 → 外層盒橘色，PTP=50")
            elif ((not in_blue) and self.slow_mode) or self.paused:
                self.slow_mode = False
                self.tcp_points = self.tcp_points_fast
                self.sim.update_slow_box_color(color=SLOW_BOX_COLOR, line_width=SLOW_BOX_LINE_WIDTH)
                self.get_logger().info("ℹ️ 離開減速區 → 外層盒藍色，PTP=100")

    def send_ptp(self, script: str):
        if not self.send_cli.service_is_ready():
            self.get_logger().warning("/send_script not ready")
            if not self.send_cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().error("無法連接 /send_script，跳過")
                return
        req = SendScript.Request()
        req.id='demo'
        req.script=script
        self.send_cli.call_async(req)

    def send_stop_and_clear(self):
        if not self.send_cli.service_is_ready():
            return
        req = SendScript.Request()
        req.id='clear'
        req.script='StopAndClearBuffer()'
        self.send_cli.call_async(req)

    def feedback_cb(self, msg: FeedbackState):
        try:
            q = np.asarray(list(msg.joint_pos)[:6], dtype=float)
            if np.max(np.abs(q)) > (np.pi*1.1):
                q = np.deg2rad(q)
            anchor = np.array([0,0,np.pi/2,0,np.pi/2,0], dtype=float)
            if np.all(np.abs(q - anchor) < 0.01):
                return
            if not hasattr(self, "_last_q_accept"):
                self._last_q_accept = q
            if np.linalg.norm(q - self._last_q_accept, ord=np.inf) < 1e-3:
                return
            self._last_q_accept = q
            self.sim.update_joint_angles(q.tolist())
        except Exception as e:
            self.get_logger().error(f"Feedback 錯誤: {e}")

    def run_loop(self):
        idx = 0
        while rclpy.ok() and not self._shutdown:
            while True:
                if self._shutdown:
                    break
                with self.lock:
                    if not self.paused:
                        break
                time.sleep(0.01)
            if self._shutdown:
                break

            with self.lock:
                if self.slow_mode:
                    self.send_stop_and_clear()
                    script = self.tcp_points_slow[idx]
                else:
                    script = self.tcp_points_fast[idx]

            self.send_ptp(script)

            t0 = time.time()
            while time.time() - t0 < 3 and not self._shutdown:
                with self.lock:
                    if self.paused:
                        break
                time.sleep(0.01)

            idx = (idx + 1) % len(self.tcp_points_fast)

    def shutdown(self):
        self._shutdown = True


# =========================
# Calibration loader for a pair
# =========================
def _build_stereo_F_and_extrinsics(calib_l, calib_r, pair_folder):
    chessboard = (4,3)
    square_size=71.0
    objp = np.zeros((chessboard[0]*chessboard[1],3), np.float32)
    objp[:,:2] = square_size*np.mgrid[0:chessboard[0],0:chessboard[1]].T.reshape(-1,2)

    L_imgs = index_by_id(glob.glob(f"{CALIB_ROOT}/{pair_folder}/img_l/*.png"))
    R_imgs = index_by_id(glob.glob(f"{CALIB_ROOT}/{pair_folder}/img_r/*.png"))
    ids = sorted(set(L_imgs.keys()) & set(R_imgs.keys()))
    if not ids:
        raise RuntimeError(f"{pair_folder} 沒有共同ID可配對")

    obj_pts, img_pts_l, img_pts_r = [], [], []
    for k in ids:
        a=cv2.imread(L_imgs[k])
        b=cv2.imread(R_imgs[k])
        if a is None or b is None:
            continue
        g1=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        g2=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
        r1,c1=cv2.findChessboardCorners(g1,chessboard,None)
        r2,c2=cv2.findChessboardCorners(g2,chessboard,None)
        if r1 and r2:
            obj_pts.append(objp.copy())
            img_pts_l.append(c1)
            img_pts_r.append(c2)

    if len(obj_pts) < 3:
        raise RuntimeError(f"{pair_folder} 標定影像太少")

    _,_,_,_,_, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, img_pts_l, img_pts_r,
        calib_l['K'], calib_l['D'],
        calib_r['K'], calib_r['D'],
        calib_l['image_size'],
        criteria=(cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS,100,1e-5),
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    return F

def _cam_to_3x4(T_c_base):
    return np.hstack([T_c_base[:3,:3], T_c_base[:3,3].reshape(3,1)]).astype(np.float64)

def load_pair_config(pair_name, pair_folder, cam_name_L, cam_name_R, T_offset_use, apply_align, align_xy):
    calib_l = calibrate_camera_from_folder(f"{CALIB_ROOT}/{pair_folder}/img_l")
    calib_r = calibrate_camera_from_folder(f"{CALIB_ROOT}/{pair_folder}/img_r")

    F = _build_stereo_F_and_extrinsics(calib_l, calib_r, pair_folder)

    # 依你原本邏輯：固定 Rz180 乘上 T_offset
    Rz180 = np.array([[-1.0,0.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,1.0]], dtype=np.float64)
    Tz = np.eye(4, dtype=np.float64)
    Tz[:3,:3] = Rz180
    T_offset_use = reorthonormalize_transform(Tz @ T_offset_use)

    T_chess_cl = get_homogeneous_transform(calib_l['R_chess_cam'], calib_l['t_chess_cam'])
    T_chess_cr = get_homogeneous_transform(calib_r['R_chess_cam'], calib_r['t_chess_cam'])

    T_base_cl = reorthonormalize_transform(T_offset_use @ T_chess_cl)
    T_base_cr = reorthonormalize_transform(T_offset_use @ T_chess_cr)

    T_cl_base = inverse_T(T_base_cl)  # base -> camL
    T_cr_base = inverse_T(T_base_cr)  # base -> camR

    P_l = _cam_to_3x4(T_cl_base)
    P_r = _cam_to_3x4(T_cr_base)

    return PairConfig(
        pair_name=pair_name,
        cam_name_L=cam_name_L,
        cam_name_R=cam_name_R,
        camL=calib_l,
        camR=calib_r,
        P_L=P_l,
        P_R=P_r,
        T_L_base=T_cl_base,
        T_R_base=T_cr_base,
        F=F,
        apply_align=apply_align,
        align_xy=align_xy
    )


# =========================
# Camera open
# =========================
def _open_live(idx, size, fps):
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise SystemExit(f"[FATAL] 相機 /dev/video{idx} 無法開啟")
    return cap


# =========================
# Main
# =========================
def main():
    if (not USE_PAIR12) and (not USE_PAIR34):
        raise SystemExit("[FATAL] USE_PAIR12/USE_PAIR34 至少要開一個")

    # === T_offset（請填入你實際的 pair12 / pair34）===
    # pair34：沿用你提供的範例
    T_offset_12 = np.array(
        [[0.5000, 0.0000, 0.8660, 585.5224],
         [-0.8660, -0.0000, 0.5000, -44.7317],
         [0.0000, -1.0000, 0.0000, 771.0000],
         [0.0000, 0.0000, 0.0000, 1.0000]],
        dtype=np.float64
    )

    T_offset_34 = np.array(
        [[ 0.5000, -0.0000, -0.8660,  70.9776],
         [ 0.8660,  0.0000,  0.5000, 239.7317],
         [ 0.0000, -1.0000,  0.0000, 1071.0000],
         [ 0.0000,  0.0000,  0.0000,   1.0000]],
        dtype=np.float64
    )

    # === 載入 pair configs ===
    pair_cfgs = {}

    # cam 名稱只是內部 key；不影響 /dev/video 的 idx
    if USE_PAIR12:
        pair_cfgs["pair12"] = load_pair_config(
            pair_name="pair12",
            pair_folder=PAIR12_FOLDER,
            cam_name_L="cam1",
            cam_name_R="cam2",
            T_offset_use=T_offset_12,
            apply_align=APPLY_ALIGN_PAIR12,
            align_xy=ALIGN_12_XY
        )

    if USE_PAIR34:
        pair_cfgs["pair34"] = load_pair_config(
            pair_name="pair34",
            pair_folder=PAIR34_FOLDER,
            cam_name_L="cam3",
            cam_name_R="cam4",
            T_offset_use=T_offset_34,
            apply_align=APPLY_ALIGN_PAIR34,
            align_xy=ALIGN_34_XY
        )

    # === 打開需要的相機（依啟用 pair 決定）===
    need_cams = set()
    if USE_PAIR12:
        need_cams.update(["cam1", "cam2"])
    if USE_PAIR34:
        need_cams.update(["cam3", "cam4"])

    cam_name_to_idx = {
        "cam1": CAM1_IDX,
        "cam2": CAM2_IDX,
        "cam3": CAM3_IDX,
        "cam4": CAM4_IDX,
    }

    caps = {}
    for cn in sorted(need_cams):
        caps[cn] = _open_live(cam_name_to_idx[cn], LIVE_SIZE, TARGET_FPS)

    # === 模型初始化 ===
    print(f"Loading YOLO from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    has_cuda = torch.cuda.is_available()
    torch_device = torch.device("cuda:0" if has_cuda else "cpu")
    yolo_device_arg = 0 if has_cuda else "cpu"

    try:
        model.to(torch_device)
    except Exception:
        pass
    try:
        model.fuse()
    except Exception:
        pass
    if USE_GPU_HALF and has_cuda:
        try:
            model.model.half()
        except Exception:
            pass

    # ReID init（用 torch.device）
    init_reid(torch_device)

    # === 啟動 ===
    sim = TMSimulator()
    rclpy.init()
    node = MultiHumanSafetyNode(
        sim=sim,
        pair_cfgs=pair_cfgs,
        yolo_pack=(model, yolo_device_arg, torch_device),
        caps=caps
    )

    def handle_sigint(sig,frame):
        node.get_logger().info("收到中斷訊號，關閉…")
        node.shutdown()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            sim.step()
    finally:
        node.shutdown()
        time.sleep(0.1)
        node.destroy_node()
        rclpy.shutdown()
        for cap in caps.values():
            try:
                cap.release()
            except Exception:
                pass

if __name__ == '__main__':
    main()
