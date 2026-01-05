#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, re, cv2, time, glob, math, threading, signal, json
import numpy as np
import pybullet as p
import pybullet_data

# ====== YOLO / Torch ======
import torch
from ultralytics import YOLO

# ====== ROS2 ======
import rclpy
from rclpy.node import Node
from tm_msgs.srv import SendScript, SetEvent
from tm_msgs.msg import FeedbackState
from std_msgs.msg import Float32  # <--- 新增這行

from scipy.optimize import linear_sum_assignment
from datetime import datetime


CALIB_ROOT = "/home/an/tm_ws/light/"
MODEL_PATH = "/home/an/tm_ws/light/human.pt"


CAM_LINUX_INDEX = {1: 0, 2: 2, 3: 4, 4: 6}

# stereo 開關
USE_PAIR12 = False   
USE_PAIR34 = True   


LIVE_SIZE       = (1920, 1080)
TARGET_FPS      = 30.0
DETECT_IMGSZ    = 384
USE_GPU_HALF    = True
BATCH_YOLO      = True



MAX_PERSONS     = 100
COLORS3D        = [(0.0, 0.0, 0.0)] * MAX_PERSONS   

# =========================
# 高低通門檻
# =========================
# 身高門檻（mm）
MAX_TOTAL_HEIGHT_MM   = 2400.0  # 2.0 m
MIN_TOTAL_HEIGHT_MM   = 1400.0  # 1.0 m

# 中心高度門檻（mm）
MAX_CENTER_Z_MM       = 900.0
MIN_center_z_shoulder = 500.0
MAX_center_z_shoulder = 900.0


TRACK3D_MAX_AGE       = 90       
TRACK3D_DIST_THRESH   = 500.0    
PREDICT_DRAW_MAX_AGE  = 10      

# YOLO 偵測門檻
DETECT_BOX_CONF       = 0.35

# ——pair34 對齊——
APPLY_ALIGN_34 = True
ALIGN_34_XY = np.array([-400.0, 0.0], dtype=np.float64)  # 單位：mm


SKELETON_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (5,11),(6,12)
]


FLOOR_TEXTURE_PATH = "/home/an/tm_ws/light/images.png"
PLATFORM_SIZE_M    = 0.20
PLATFORM_HEIGHT_M  = 0.69
FLOOR_Z            = -0.69

AABB_MARGIN_M      = 0.05
EMA_ALPHA          = 0.25
AABB_SCALE         = 1.5 #急停區包圍盒大小（倍率）

SLOW_BOX_CENTER       = [0.0, 0.0, 0.0]
SLOW_BOX_HALF_EXTENT_M = 2.0
SLOW_BOX_COLOR        = [0, 0, 1]
SLOW_BOX_ALERT        = [1.0, 0.5, 0.0]
SLOW_BOX_LINE_WIDTH   = 4

OVERLAY_LINE_WIDTH    = 15
BBOX_LINE_WIDTH       = 2


USE_XY_SQUARE_MASK = True
XY_MASK_CENTER_M   = (0.0, 0.0)
XY_MASK_HALF_M     = 2.5
XY_MASK_Z_RANGE_M  = None   

# —— BIAS（m）——
BIAS_XY_M = (0, +0.75)
BIAS_MM   = np.array([BIAS_XY_M[0]*1000.0, BIAS_XY_M[1]*1000.0, 0.0], dtype=np.float64)

# 暫停區冷卻時間（秒）
RESUME_COOLDOWN_SEC = 0.5


_id_re = re.compile(r'(\d+)')

def extract_id(path):
    m = _id_re.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def align34_xy(center_mm: np.ndarray, X_full_mm: np.ndarray):

    if center_mm is not None and np.all(np.isfinite(center_mm)):
        center_mm = center_mm.copy()
        center_mm[0] += ALIGN_34_XY[0]
        center_mm[1] += ALIGN_34_XY[1]
    if X_full_mm is not None and X_full_mm.size > 0:
        X_full_mm = X_full_mm.copy()
        good = np.isfinite(X_full_mm).all(axis=1)
        X_full_mm[good, 0] += ALIGN_34_XY[0]
        X_full_mm[good, 1] += ALIGN_34_XY[1]
    return center_mm, X_full_mm

def index_by_id(paths):
    idx, dup = {}, {}
    for pth in paths:
        k = extract_id(pth)
        if k is None:
            continue
        if k in idx:
            dup.setdefault(k, []).append(pth)
        else:
            idx[k] = pth
    if dup:
        raise RuntimeError(f"發現重複ID: {sorted(dup.items())}")
    return idx

def get_homogeneous_transform(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3,3] = t.reshape(3)
    return T

def inverse_T(T):
    R = T[:3,:3]
    t = T[:3,3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3,:3] = R.T
    Ti[:3,3] = -R.T @ t
    return Ti

def reorthonormalize_transform(T):
    R, t = T[:3,:3], T[:3,3]
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    Tn = np.eye(4)
    Tn[:3,:3] = Rn
    Tn[:3,3] = t
    return Tn

def calibrate_camera_from_folder(folder, chessboard_size=(4,3), square_size=71.0):
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
    objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
    objp[:,:2] = square_size*np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

    objpoints, imgpoints = [], []
    images = sorted(glob.glob(os.path.join(folder,'*.png')))
    if not images:
        raise RuntimeError(f"找不到標定資料夾：{folder}")
    gray_sz = None
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(g, chessboard_size, None)
        if not ret:
            continue
        corners2 = cv2.cornerSubPix(g, corners, (11,11), (-1,-1), criteria)
        objpoints.append(objp.copy())
        imgpoints.append(corners2)
        gray_sz = g.shape[::-1]

    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_sz, None, None)
    if not ret:
        raise RuntimeError("相機標定失敗")
    R_obj_cam, _ = cv2.Rodrigues(rvecs[0])
    t_obj_cam = tvecs[0].reshape(3,1)
    R_chess_cam = R_obj_cam.T
    t_chess_cam = (-R_obj_cam.T @ t_obj_cam).reshape(3,)
    return {'K':K,'D':D,'image_size':gray_sz,'R_chess_cam':R_chess_cam,'t_chess_cam':t_chess_cam}

def undistort_points_xy(kp, K, D):
    pts = kp.reshape(-1,1,2).astype(np.float32)
    return cv2.undistortPoints(pts, K, D).reshape(-1,2)

def cam_to_base_3x4(T_c_b):
    return np.hstack([T_c_b[:3,:3], T_c_b[:3,3].reshape(3,1)]).astype(np.float64)


def dets_from_res_scaled(res, sx=1.0, sy=1.0):
    dets = []
    if (res is None) or (not hasattr(res, 'boxes')) or len(res.boxes) == 0:
        return dets

    boxes = res.boxes.xyxy.detach().cpu().numpy().copy()
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy

    has_kp = hasattr(res, "keypoints") and res.keypoints is not None
    kps = res.keypoints.xy.detach().cpu().numpy().copy() if has_kp else None
    if kps is not None:
        kps[:, :, 0] *= sx
        kps[:, :, 1] *= sy
    kpc = res.keypoints.conf.detach().cpu().numpy().copy() if has_kp else None

    bconfs = res.boxes.conf.detach().cpu().numpy()

    for i in range(len(boxes)):
        if bconfs[i] < DETECT_BOX_CONF:
            continue
        dets.append({
            "kp":      kps[i] if kps is not None else None,
            "kp_conf": kpc[i] if kpc is not None else None,
            "box":     boxes[i],
            "score":   float(bconfs[i]),
        })

    return dets


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
            for ii, k in enumerate(vi):
                if k < 17:
                    X_full[k] = Xw[ii]

            def _ok(idx):
                return (idx is not None) and (0<=idx<17) and np.all(np.isfinite(X_full[idx]))
            if _ok(11) and _ok(12):
                center = 0.5*(X_full[11]+X_full[12])
            elif _ok(5) and _ok(6):
                center = 0.5*(X_full[5]+X_full[6])
            elif _ok(0):
                center = X_full[0]

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
    height_mm = float(z_vals.max() - z_vals.min())
    return height_mm

def estimate_center_height_mm_from_shoulders(X_full_mm):
    if X_full_mm is None or X_full_mm.shape[0] < 7:
        return None
    p5 = X_full_mm[5]
    p6 = X_full_mm[6]
    if not (np.all(np.isfinite(p5)) and np.all(np.isfinite(p6))):
        return None
    center_z = float(0.5 * (p5[2] + p6[2]))
    return center_z

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
        center_z = float((lowest2.sum() + highest2.sum()) / 4.0)
    else:
        center_z = float(0.5 * (z_sorted.min() + z_sorted.max()))
    return center_z

def tri_from_pairs(
    pairs, dL, dR,
    K1, D1, K2, D2,
    P1n, P2n, src_tag
):

    det3d_list = []
    for (iL, iR, cost) in pairs:
        c3d, X = triangulate_center_simple(
            dL[iL], dR[iR], K1, D1, K2, D2, P1n, P2n
        )
        if c3d is None:
            # print(f"[DEBUG] DROP tri pair src={src_tag} (iL={iL}, iR={iR}) reason=no_center")
            continue

        
        
        if src_tag == '34' and APPLY_ALIGN_34:
            c3d, X = align34_xy(c3d, X)


        
        height_mm = estimate_total_height_mm(X)

        
        center_z_shoulder = estimate_center_height_mm_from_shoulders(X)
        center_z_ext      = None
        center_z_used     = None   
        cz_source         = "raw"  

        
        if height_mm is not None:
            if (height_mm < MIN_TOTAL_HEIGHT_MM) or (height_mm > MAX_TOTAL_HEIGHT_MM):
                # print(f"[DEBUG] DROP tri pair src={src_tag} (iL={iL}, iR={iR}) "
                #       f"reason=height_out_of_range height_mm={height_mm:.1f}")
                continue

        
        c3d = np.asarray(c3d, dtype=np.float64)

        if center_z_shoulder is not None:
            if (center_z_shoulder < MIN_center_z_shoulder) or (center_z_shoulder > MAX_center_z_shoulder):
                # print(f"[DEBUG] DROP tri pair src={src_tag} (iL={iL}, iR={iR}) "
                #       f"reason=shoulder_center_z_out_of_range center_z={center_z_shoulder:.1f}")
                continue
            c3d[2] = center_z_shoulder
            center_z_used = center_z_shoulder
            cz_source = "shoulder"
        else:
            center_z_ext = estimate_center_height_mm_from_extremes(X)
            if center_z_ext is not None:
                if center_z_ext > MAX_CENTER_Z_MM:
                    # print(f"[DEBUG] DROP tri pair src={src_tag} (iL={iL}, iR={iR}) "
                    #       f"reason=extremes_center_z_too_high center_z={center_z_ext:.1f}")
                    continue
                c3d[2] = center_z_ext
                center_z_used = center_z_ext
                cz_source = "extremes"
            else:
                cz_source = "none"

        
        h_str       = f"{height_mm:.1f}"          if height_mm is not None else "None"
        cz_sh_str   = f"{center_z_shoulder:.1f}"  if center_z_shoulder is not None else "None"
        cz_ext_str  = f"{center_z_ext:.1f}"       if center_z_ext is not None else "None"


        # print(
        #     f"[DEBUG] KEEP tri pair src={src_tag} (iL={iL}, iR={iR}) "
        #     f"height_mm={h_str}, "
        #     f"center_z_shoulder_mm={cz_sh_str}, "
        #     f"center_z_ext_mm={cz_ext_str}, "
        # )

        if src_tag == '34':
            boxes = {3: dL[iL]["box"].copy(),
                    4: dR[iR]["box"].copy()}
            cams  = {3: iL, 4: iR}
        else:  # '12'
            boxes = {1: dL[iL]["box"].copy(),
                    2: dR[iR]["box"].copy()}
            cams  = {1: iL, 2: iR}


        det3d_list.append({
            "center": np.asarray(c3d, dtype=np.float64),
            "X_full": np.asarray(X, dtype=np.float64),
            "src":    src_tag,
            "boxes":  boxes,
            "cams":   cams,
        })

    return det3d_list



def stereo_triangulate_bruteforce(
    dL, dR,
    K1, D1, K2, D2,
    P1n, P2n,
    src_tag
):

    n1, n2 = len(dL), len(dR)
    if n1 == 0 or n2 == 0:
        return []
    pairs = [(iL, iR, 0.0) for iL in range(n1) for iR in range(n2)]
    det3d_list = tri_from_pairs(
        pairs, dL, dR,
        K1, D1, K2, D2,
        P1n, P2n,
        src_tag
    )
    return det3d_list


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
            [0,0,0,1,0,0 ],
            [0,0,0,0,1,0 ],
            [0,0,0,0,0,1 ]
        ], dtype=np.float64)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        H = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,0,0,0]
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
        self.src = det3d["src"]
        self.boxes = det3d["boxes"].copy()
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
        self.src = det3d["src"]
        self.boxes = det3d["boxes"].copy()
        self.time_since_update = 0

class MultiObjectTracker3D:
    def __init__(self,
                 max_age=TRACK3D_MAX_AGE,
                 dist_thresh=TRACK3D_DIST_THRESH):
        self.max_age = max_age
        self.dist_thresh = dist_thresh
        self.tracks = []
        self._next_id = 1

    def predict(self, dt=1.0):
        for t in self.tracks:
            t.predict(dt)

    def update(self, det3d_list):
        nD = len(det3d_list)
        if len(self.tracks) == 0:
            for d in det3d_list:
                tr = Track3D(self._next_id, d)
                self._next_id += 1
                self.tracks.append(tr)
        else:
            nT = len(self.tracks)
            if nD > 0:
                C = np.zeros((nT, nD), dtype=np.float64)
                for i, tr in enumerate(self.tracks):
                    p = tr.center
                    for j, d in enumerate(det3d_list):
                        c = d["center"]
                        C[i,j] = np.linalg.norm(p - c)

                rows, cols = linear_sum_assignment(C)
                matched_t = set()
                matched_d = set()
                for r, c in zip(rows, cols):
                    dist = C[r,c]
                    if dist <= self.dist_thresh:
                        self.tracks[r].update(det3d_list[c])
                        matched_t.add(r)
                        matched_d.add(c)

                for j, d in enumerate(det3d_list):
                    if j in matched_d:
                        continue
                    tr = Track3D(self._next_id, d)
                    self._next_id += 1
                    self.tracks.append(tr)

            new_tracks = []
            for tr in self.tracks:
                if tr.time_since_update <= self.max_age:
                    new_tracks.append(tr)
            self.tracks = new_tracks

        active_tracks = [
            t for t in self.tracks
            if t.time_since_update <= PREDICT_DRAW_MAX_AGE
        ]
        return active_tracks


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

        self.tm5_id = p.loadURDF("/home/an/tm_ws/src/tmr_ros2/tm_description/urdf/tm5-900.urdf",
                                 basePosition=[0,0,0], useFixedBase=True)
        nj = p.getNumJoints(self.tm5_id)
        p.changeVisualShape(self.tm5_id, -1, rgbaColor=[1,1,1,1])
        for link_idx in range(nj):
            p.changeVisualShape(self.tm5_id, link_idx, rgbaColor=[1,1,1,1])

        self.last_green_exit = None

        

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
        for t in range(MAX_PERSONS):
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
            ([xM,yM,zm],[xm,yM,zm]), ([xm,yM,zm],[xm,ym,zm]),
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

        if pts3d_m is None or valid_mask is None:
          
            self.clear_skeleton_slot(slot_idx)
            return

        if not (0 <= slot_idx < len(self.skel_line_ids)):
            return

        line_set = self.skel_line_ids[slot_idx]
        for k,(i,j) in enumerate(SKELETON_EDGES):
            is_valid = (
                i < len(valid_mask) and j < len(valid_mask)
                and valid_mask[i] and valid_mask[j]
            )
            if is_valid:
                p1 = pts3d_m[i].tolist()
                p2 = pts3d_m[j].tolist()
                p.addUserDebugLine(
                    p1, p2,
                    lineColorRGB=color_rgb,
                    lineWidth=4,              
                    lifeTime=0,
                    replaceItemUniqueId=line_set[k]
                )
            else:
                
                p.addUserDebugLine(
                    [0,0,0], [0,0,0],
                    lineColorRGB=color_rgb,
                    lineWidth=1,             
                    lifeTime=0,
                    replaceItemUniqueId=line_set[k]
                )
            self.prev_valid[slot_idx][k] = is_valid


    def clear_skeleton_slot(self, slot_idx):
        if not (0 <= slot_idx < len(self.skel_line_ids)):
            return
        for lid in self.skel_line_ids[slot_idx]:
            p.addUserDebugLine(
                [0,0,0],[0,0,0],
                [0,0,0],
                1,0,
                replaceItemUniqueId=lid
            )
        self.prev_valid[slot_idx] = [False]*len(SKELETON_EDGES)

    def step(self):
        p.stepSimulation()
        time.sleep(1.0/240)


class MultiHumanSafetyNode(Node):
    def __init__(self, sim, calib, proj, yolo_pack):
        super().__init__('multi_human_safety_3cams')
        self.sim = sim

        # 相機內參
        self.cam1, self.cam2, self.cam3, self.cam4 = calib

        # 投影矩陣（base 座標系）
        (self.P1n_12, self.P2n_12,
        self.P3n_34, self.P4n_34) = proj

        # YOLO
        self.model, self.device = yolo_pack

        # ROS2 services
        self.send_cli = self.create_client(SendScript, '/send_script')
        while not self.send_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning("Waiting for /send_script...")
        self.event_cli = self.create_client(SetEvent, '/set_event')
        while not self.event_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning("Waiting for /set_event...")
        self.create_subscription(FeedbackState, '/feedback_states', self.feedback_cb, 10)

        # PTP 路徑
        self.tcp_points_fast = [
            'PTP("CPP",504,-107,354,-179,-45,90,100,100,100,false)',
        ]
        self.tcp_points_slow = [
            'PTP("CPP",504,-107,354,-179,-45,90,100,100,100,false)',
        ]
        self.tcp_points = self.tcp_points_fast
        self.paused = False
        self.slow_mode = False
        self.lock = threading.Lock()

        
        
        self.cap1 = self._open_live(CAM_LINUX_INDEX[1], LIVE_SIZE, TARGET_FPS) if USE_PAIR12 else None
        self.cap2 = self._open_live(CAM_LINUX_INDEX[2], LIVE_SIZE, TARGET_FPS) if USE_PAIR12 else None
        self.cap3 = self._open_live(CAM_LINUX_INDEX[3], LIVE_SIZE, TARGET_FPS) if USE_PAIR34 else None
        self.cap4 = self._open_live(CAM_LINUX_INDEX[4], LIVE_SIZE, TARGET_FPS) if USE_PAIR34 else None


        
        self.tracker3d = MultiObjectTracker3D(
            max_age=TRACK3D_MAX_AGE,
            dist_thresh=TRACK3D_DIST_THRESH
        )

        self._shutdown = False
        self.dt = 1.0 / TARGET_FPS if TARGET_FPS > 0 else 1.0

        
        threading.Thread(target=self.detect_loop, daemon=True).start()

        
        sb = self.sim.get_slow_bounds()
        if sb:
            xm,xM,ym,yM,zm,zM = sb
            self.get_logger().info(
                f"固定減速盒：中心(0,0,0)，半徑={SLOW_BOX_HALF_EXTENT_M} m，"
                f"x∈[{xm:.2f},{xM:.2f}], y∈[{ym:.2f},{yM:.2f}], z∈[{zm:.2f},{zM:.2f}]"
            )
        # ★ 新增：速度比例 Publisher
        self.speed_ratio_pub = self.create_publisher(Float32, '/safety/speed_ratio', 10)
        
        # threading.Thread(target=self.run_loop, daemon=True).start()

    def _open_live(self, idx, size, fps):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        cap.set(cv2.CAP_PROP_FPS, fps)
        if not cap.isOpened():
            raise SystemExit(f"[FATAL] 相機 /dev/video{idx} 無法開啟")
        return cap


    # YOLO resize helper
    def _resize(self, frame):

        h, w = frame.shape[:2]
        if DETECT_IMGSZ is None or h == DETECT_IMGSZ:
            return frame, 1.0, 1.0
        scale = DETECT_IMGSZ / float(h)
        new_w = int(round(w * scale))
        resized = cv2.resize(frame, (new_w, DETECT_IMGSZ), interpolation=cv2.INTER_LINEAR)
        sx = w / float(new_w)
        sy = h / float(DETECT_IMGSZ)
        return resized, sx, sy

    def _infer_yolo(self, frames_resized):

        if not frames_resized:
            return []

        if BATCH_YOLO:
            with torch.no_grad():
                results = self.model(frames_resized, imgsz=DETECT_IMGSZ, verbose=False)
            
            return list(results)
        else:
            outs = []
            with torch.no_grad():
                for img in frames_resized:
                    r = self.model(img, imgsz=DETECT_IMGSZ, verbose=False)
                    outs.append(r[0])
            return outs

    def _apply_xy_square_mask(self, center_mm):

        if not USE_XY_SQUARE_MASK:
            return True

        x_m = center_mm[0] / 1000.0
        y_m = center_mm[1] / 1000.0
        z_m = center_mm[2] / 1000.0

        cx, cy = XY_MASK_CENTER_M
        half = float(XY_MASK_HALF_M)

        inside_xy = (cx - half <= x_m <= cx + half) and (cy - half <= y_m <= cy + half)
        if not inside_xy:
            return False

        if XY_MASK_Z_RANGE_M is not None:
            zmin, zmax = XY_MASK_Z_RANGE_M
            if not (zmin <= z_m <= zmax):
                return False

        return True

    # ★ 新增：通用送腳本函式 (移植自舊程式)
    def _send_script(self, script_str):
        if not self.send_cli.service_is_ready():
            self.get_logger().warn("SendScript service not ready")
            return
        req = SendScript.Request()
        req.id = "safety_override" # 隨意取名
        req.script = script_str
        self.send_cli.call_async(req)

    # ★ 新增：發布給 switch20_20.py 看的比例
    def _pub_ratio(self, val: float):
        msg = Float32()
        msg.data = float(val)
        self.speed_ratio_pub.publish(msg)

    # ★ 新增：發布給手臂硬體的強制減速指令
    def send_speed_override(self, percent: int):
        pct = max(1, min(100, int(percent)))
        script = f"SpeedOverride({pct})"
        self.get_logger().info(f"[SAFETY] SpeedOverride -> {pct}%")
        self._send_script(script)

    def _send_pause(self):
        
        if not self.event_cli.service_is_ready():
            self.get_logger().warn("SetEvent service not ready (Pause)")
            return
        req = SetEvent.Request()
        req.func = SetEvent.Request.PAUSE  
        req.arg0 = 0
        req.arg1 = 0
        self.event_cli.call_async(req)
        self.get_logger().warn("[SAFETY] Send Pause")

    def _send_resume(self):
        
        if not self.event_cli.service_is_ready():
            self.get_logger().warn("SetEvent service not ready (Resume)")
            return
        req = SetEvent.Request()
        req.func = SetEvent.Request.RESUME  
        req.arg0 = 0
        req.arg1 = 0
        self.event_cli.call_async(req)
        self.get_logger().info("[SAFETY] Send Resume")


    def _update_safety_state(self, active_tracks):
        now = time.time()
        any_in_green = False
        any_in_slow = False

        
        try:
            xm, xM, ym, yM, zm, zM = self.sim.current_bounds
        except Exception:
            xm = xM = ym = yM = zm = zM = None

        
        slow_bounds = self.sim.get_slow_bounds()
        if slow_bounds:
            sxm, sxM, sym, syM, szm, szM = slow_bounds
        else:
            sxm = sxM = sym = syM = szm = szM = None

        
        for tr in active_tracks:
            X_full_mm = tr.X_full_mm
            if X_full_mm is None:
                continue

            pts3d_m = X_full_mm.astype(np.float64) / 1000.0
            valid = np.isfinite(pts3d_m).all(axis=1)
            valid_pts = pts3d_m[valid]

            
            if xm is not None and not any_in_green:
                for (x, y, z) in valid_pts:
                    if xm <= x <= xM and ym <= y <= yM and zm <= z <= zM:
                        any_in_green = True
                        break


            if slow_bounds and not any_in_slow:
                for (x, y, z) in valid_pts:
                    if sxm <= x <= sxM and sym <= y <= syM and szm <= z <= szM:
                        any_in_slow = True
                        break


        if any_in_green:
            self.sim.last_green_exit = None
            if not self.paused:
                self.paused = True
                self._send_pause()
                self.sim.update_bounding_box_color(color=[1, 0, 0], line_width=BBOX_LINE_WIDTH)
                self.sim.show_overlay_from_current(color=[1, 0, 0], line_width=OVERLAY_LINE_WIDTH)
        else:

            if self.sim.last_green_exit is None:
                self.sim.last_green_exit = now
            

            if self.paused and (now - self.sim.last_green_exit >= RESUME_COOLDOWN_SEC):
                self.paused = False
                self._send_resume()
                self.sim.update_bounding_box_color(color=[0, 1, 0], line_width=BBOX_LINE_WIDTH)
                self.sim.hide_overlay()


        if any_in_slow:
            self.sim.last_slow_exit = None 
            if not self.slow_mode:
                self.slow_mode = True
                
                # 若您希望跟來源檔案一樣完全依賴硬體減速，可以註解掉下面這行切換點位的程式
                # self.tcp_points = self.tcp_points_slow 
                
                self.sim.update_slow_box_color(color=SLOW_BOX_ALERT, line_width=SLOW_BOX_LINE_WIDTH)
                self.get_logger().info("[SAFETY] Enter SLOW mode")

                # ★★★ 請加入這兩行 (觸發減速) ★★★
                self._pub_ratio(0.0001)       # 通知 python 迴圈 (switch20_20)
                self.send_speed_override(5)   # 通知手臂硬體降速至 5%

        else:
            
            if self.slow_mode:
                
                if self.sim.last_slow_exit is None:
                    self.sim.last_slow_exit = now
                
                # 過了冷卻時間，恢復速度
                if (now - self.sim.last_slow_exit >= RESUME_COOLDOWN_SEC):
                    self.slow_mode = False
                    
                    # 若上面註解了，這裡也可以註解掉
                    # self.tcp_points = self.tcp_points_fast
                    
                    self.sim.update_slow_box_color(color=SLOW_BOX_COLOR, line_width=SLOW_BOX_LINE_WIDTH)
                    self.get_logger().info("[SAFETY] Back to FAST mode")
                    self.sim.last_slow_exit = None

                    # ★★★ 請加入這兩行 (恢復全速) ★★★
                    self._pub_ratio(1.0)          # 通知 python 迴圈恢復
                    self.send_speed_override(100) # 通知手臂硬體恢復 100%

    def feedback_cb(self, msg: FeedbackState):

        try:
            
            angles = list(msg.joint_angle[:6])
        except AttributeError:
            
            angles = list(msg.joint_pos[:6])
        self.sim.update_joint_angles(angles)

    def send_ptp(self, script: str):

        if not self.send_cli.service_is_ready():
            self.get_logger().warning("/send_script not ready")

            if not self.send_cli.wait_for_service(timeout_sec=2.0):
                self.get_logger().error("無法連接 /send_script，跳過本次 PTP")
                return

        req = SendScript.Request()

        if hasattr(req, "id"):
            req.id = "demo"
        req.script = script

        self.send_cli.call_async(req)
        # self.get_logger().info(f" 已送出 PTP 指令: {script}")

    def send_stop_and_clear(self):

        if not self.send_cli.service_is_ready():
            self.get_logger().warning("/send_script not ready (StopAndClearBuffer)")
            return

        req = SendScript.Request()
        if hasattr(req, "id"):
            req.id = "clear"
        req.script = "StopAndClearBuffer()"

        self.send_cli.call_async(req)
        self.get_logger().info(" 已送出 StopAndClearBuffer()")


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



    def detect_loop(self):


        while not self._shutdown:

            frame1 = frame2 = frame3 = frame4 = None

            # pair12
            if self.cap1 is not None:
                ret1, frame1 = self.cap1.read()
                if not ret1:
                    frame1 = None
            if self.cap2 is not None:
                ret2, frame2 = self.cap2.read()
                if not ret2:
                    frame2 = None

            # pair34
            if self.cap3 is not None:
                ret3, frame3 = self.cap3.read()
                if not ret3:
                    frame3 = None
            if self.cap4 is not None:
                ret4, frame4 = self.cap4.read()
                if not ret4:
                    frame4 = None


            need_12_ok = (not USE_PAIR12) or (frame1 is not None and frame2 is not None)
            need_34_ok = (not USE_PAIR34) or (frame3 is not None and frame4 is not None)
            if not (need_12_ok and need_34_ok):
                self.get_logger().warning("Camera read failed, retrying...")
                time.sleep(0.05)
                continue



            imgs_resized = []
            scales = []
            cam_ids = []  

            # cam1
            if self.cap1 is not None and frame1 is not None:
                r, sx, sy = self._resize(frame1)
                imgs_resized.append(r); scales.append((sx, sy)); cam_ids.append(1)

            # cam2
            if self.cap2 is not None and frame2 is not None:
                r, sx, sy = self._resize(frame2)
                imgs_resized.append(r); scales.append((sx, sy)); cam_ids.append(2)

            # cam3
            if self.cap3 is not None and frame3 is not None:
                r, sx, sy = self._resize(frame3)
                imgs_resized.append(r); scales.append((sx, sy)); cam_ids.append(3)

            # cam4
            if self.cap4 is not None and frame4 is not None:
                r, sx, sy = self._resize(frame4)
                imgs_resized.append(r); scales.append((sx, sy)); cam_ids.append(4)

            results = self._infer_yolo(imgs_resized)


            dets_cam = {1: [], 2: [], 3: [], 4: []}
            for k, cam_id in enumerate(cam_ids):
                sx, sy = scales[k]
                dets_cam[cam_id] = dets_from_res_scaled(results[k], sx, sy)



            det3d_all = []

            # pair12 (cam1+cam2)
            if USE_PAIR12 and (len(dets_cam[1]) > 0) and (len(dets_cam[2]) > 0):
                det12 = stereo_triangulate_bruteforce(
                    dets_cam[1], dets_cam[2],
                    self.cam1["K"], self.cam1["D"],
                    self.cam2["K"], self.cam2["D"],
                    self.P1n_12, self.P2n_12,
                    src_tag='12'
                )
                det3d_all.extend(det12)

            
            # pair34 (cam3+cam4)
            if USE_PAIR34 and (len(dets_cam[3]) > 0) and (len(dets_cam[4]) > 0):
                det34 = stereo_triangulate_bruteforce(
                    dets_cam[3], dets_cam[4],
                    self.cam3["K"], self.cam3["D"],
                    self.cam4["K"], self.cam4["D"],
                    self.P3n_34, self.P4n_34,
                    src_tag='34'
                )
                det3d_all.extend(det34)


            # merge
            merged_det3d = []
            for d in det3d_all:
                c = d["center"]
                duplicated = False
                for e in merged_det3d:
                    if np.linalg.norm(c - e["center"]) < 10.0:  # cm
                        duplicated = True
                        break
                if not duplicated:
                    merged_det3d.append(d)


            det3d_filtered = []
            for d in merged_det3d:
                c_mm = d["center"].astype(np.float64)
                X_full = d["X_full"].astype(np.float64)

                c_mm_bias = c_mm + BIAS_MM
                


                X_full_bias = X_full.copy()
                good = np.isfinite(X_full_bias).all(axis=1)
                X_full_bias[good] += BIAS_MM

                d2 = {
                    "center": c_mm_bias,
                    "X_full": X_full_bias,
                    "src":    d["src"],
                    "boxes":  d["boxes"],
                    "cams":   d["cams"],
                }
                det3d_filtered.append(d2)


            
            self.tracker3d.predict(dt=self.dt)
            active_tracks = self.tracker3d.update(det3d_filtered)
            

            
            active_tracks = sorted(active_tracks, key=lambda t: t.id)
            for slot_idx, tr in enumerate(active_tracks[:MAX_PERSONS]):
                

                if not self._apply_xy_square_mask(tr.center):
                    self.sim.clear_skeleton_slot(slot_idx)
                    continue

                X_full_mm = tr.X_full_mm
                if X_full_mm is None: 
                    self.sim.clear_skeleton_slot(slot_idx)
                    continue
                

                draw_color = list(COLORS3D[slot_idx % len(COLORS3D)])

                valid_mask = np.isfinite(X_full_mm).all(axis=1)
                if not np.any(valid_mask): 
                    self.sim.clear_skeleton_slot(slot_idx)
                    continue
                
                pts3d_m = np.nan_to_num(X_full_mm, nan=0.0) / 1000.0
                
                self.sim.draw_skeleton_slot(slot_idx, pts3d_m, valid_mask, draw_color)
            

            for slot_idx in range(len(active_tracks), MAX_PERSONS):
                self.sim.clear_skeleton_slot(slot_idx)


            self._update_safety_state(active_tracks)

        self.get_logger().info("detect_loop exit")

def main():
    rclpy.init()


    cam1_folder = os.path.join(CALIB_ROOT, "pair12", "img_l")
    cam2_folder = os.path.join(CALIB_ROOT, "pair12", "img_r")
    cam3_folder = os.path.join(CALIB_ROOT, "pair34", "img_l")
    cam4_folder = os.path.join(CALIB_ROOT, "pair34", "img_r")


    print(f"[CALIB] cam1 folder = {cam1_folder}")
    print(f"[CALIB] cam2 folder = {cam2_folder}")
    print(f"[CALIB] cam3 folder = {cam3_folder}")
    print(f"[CALIB] cam4 folder = {cam4_folder}")

    cam1 = calibrate_camera_from_folder(cam1_folder)
    cam2 = calibrate_camera_from_folder(cam2_folder)
    cam3 = calibrate_camera_from_folder(cam3_folder)
    cam4 = calibrate_camera_from_folder(cam4_folder)



    T_offset_12 = np.array(
[[0.8660, 0.0000, -0.5000, 144.7317],
 [0.5000, 0.0000, 0.8660, 435.5224],
 [0.0000, -1.0000, 0.0000, 571.0000],
 [0.0000, 0.0000, 0.0000, 1.0000]],
        dtype=np.float64
    )

    T_offset_34 = np.array(
[[-0.7660, -0.0000, -0.6428, -322.6486],
 [0.6428, -0.0000, -0.7660, 95.6827],
 [0.0000, -1.0000, 0.0000, 671.0000],
 [0.0000, 0.0000, 0.0000, 1.0000]],
        dtype=np.float64
    )


    Rz180 = np.array(
        [[-1.0, 0.0, 0.0],
         [ 0.0,-1.0, 0.0],
         [ 0.0, 0.0, 1.0]], dtype=np.float64
    )
    Tz = np.eye(4, dtype=np.float64)
    Tz[:3, :3] = Rz180

    T_offset_12_use = reorthonormalize_transform(Tz @ T_offset_12)
    T_offset_34_use = reorthonormalize_transform(Tz @ T_offset_34)


    T_chess_c1 = get_homogeneous_transform(cam1['R_chess_cam'], cam1['t_chess_cam'])
    T_chess_c2 = get_homogeneous_transform(cam2['R_chess_cam'], cam2['t_chess_cam'])
    T_chess_c3 = get_homogeneous_transform(cam3['R_chess_cam'], cam3['t_chess_cam'])
    T_chess_c4 = get_homogeneous_transform(cam4['R_chess_cam'], cam4['t_chess_cam'])

    # pair12
    T_base_c1 = reorthonormalize_transform(T_offset_12_use @ T_chess_c1)
    T_base_c2 = reorthonormalize_transform(T_offset_12_use @ T_chess_c2)

    # pair34
    T_base_c3 = reorthonormalize_transform(T_offset_34_use @ T_chess_c3)
    T_base_c4 = reorthonormalize_transform(T_offset_34_use @ T_chess_c4)

    # base->cam
    T_c1_base = inverse_T(T_base_c1)
    T_c2_base = inverse_T(T_base_c2)
    T_c3_base = inverse_T(T_base_c3)
    T_c4_base = inverse_T(T_base_c4)

    # 投影矩陣 [R|t]（base 座標系）
    P1n_12 = cam_to_base_3x4(T_c1_base)
    P2n_12 = cam_to_base_3x4(T_c2_base)
    P3n_34 = cam_to_base_3x4(T_c3_base)
    P4n_34 = cam_to_base_3x4(T_c4_base)

    calib = (cam1, cam2, cam3, cam4)
    proj  = (P1n_12, P2n_12, P3n_34, P4n_34)




    sim = TMSimulator()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[YOLO] device = {device}")
    model = YOLO(MODEL_PATH)
    model.to(device)
    if USE_GPU_HALF and device != "cpu":
        model.model.half()
    else:
        model.model.float()
    yolo_pack = (model, device)


    node = MultiHumanSafetyNode(sim, calib, proj, yolo_pack)

    print("=== 程式啟動：主迴圈開始 ===")
    try:

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)
            sim.step()  
    except KeyboardInterrupt:
        pass
    finally:
        node._shutdown = True

        if node.cap1 is not None: node.cap1.release()
        if node.cap2 is not None: node.cap2.release()
        if node.cap3 is not None: node.cap3.release()
        if node.cap4 is not None: node.cap4.release()
        cv2.destroyAllWindows()
        p.disconnect()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
