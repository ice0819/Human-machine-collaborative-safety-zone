
# Human-Machine Collaborative Safety Zone

**Human-Machine Collaborative Safety Zone**  
Multiple cameras are deployed around the workspace and combined with multi-person 3D human pose tracking to capture the spatial distances and motion relationships between each human joint and the robotic arm. The system continuously monitors whether any human skeleton enters the human–robot collaboration zone.
<div align="center">
<img width="525" height="442" alt="圖片1" src="https://github.com/user-attachments/assets/3ec22577-bc18-42d3-be0f-b6020fcfca42" />
</div>
<div align="center">
https://github.com/user-attachments/assets/09647b08-a651-4d4a-a77b-13289065649b
</div>

## Directory Structure and File Descriptions

| Category | File Name | Description |
| :--- | :--- | :--- |
| **Main Program** | `fourcamsafe.areav13.2floor.py` | **Main human–robot collaborative safety light curtain program** (uses OSNet; higher performance overhead) |
| **Backup Main Program** | `safe.areav12.4floor.py` | **Main human–robot collaborative safety light curtain program** (lighter performance overhead; more ghost tracks) |
| **Four-Camera Capture Tool** | `4capture.py` | Captures calibration photos and hand–eye pose photos using 4 cameras |
| **Two-Camera Capture Tool** | `2capture.py` | Captures calibration photos and hand–eye pose photos using 2 cameras(If you need it) |
| **Hand–Eye Coordinate Transform** | `tcp_to_base3.py` | Input the robot TCP pose to obtain the transform matrix from chessboard to robot base; paste into the main program |
| **Human Skeleton Model Weights** | `human.pt` | Used for 2D human skeleton inference; imported via program path |
| **OSNet Weights** | `osnet_x0_25_msmt17.pt` | Used for multi-person matching; imported via program path |
| **3D Simulation Floor Texture** | `images.png` | Replaces floor material for better visual appearance |
| **URDF Folder** | `tm_description` | Robot arm model for the 3D physics simulation environment |

---

## Calibration Workflow and Tools

If the camera positions or viewing angles change, recalibrate by following the steps below in order.

### 1. Collect Calibration Photos
```bash
cd (to the location of the 4capture.py folder)
```
```bash
python3 4capture.py
```

This script provides synchronized previews from four cameras and captures stereo calibration chessboard images in a “paired-camera” manner.

- Files are saved as the original PNGs (no overlays). Filenames are `left_{N}.png` / `right_{N}.png`. The save directory can be modified depending on your environment.
- `left_0.png` / `right_0.png` are the photos taken with the chessboard rigidly mounted to the robot end-effector. Record the TCP pose at that moment: `X Y Z Rx Ry Rz` (**rotation angles must be (90, 0, XX)**).
- The program scans existing filenames in the folder and automatically continues numbering to avoid overwriting.
- `CAM_SOURCES`: The four camera sources are enumerated based on USB insertion order (integer index or `/dev/videoX`). Example: `[0, 2, 4, 6]`.
- `TARGET_SIZE`: Capture resolution; must not be changed (saved images also use this resolution). Example: `(1920, 1080)`.

#### Keyboard Controls

- `1`: Switch to **pair12 (Cam1–Cam2)**
- `3`: Switch to **pair34 (Cam3–Cam4)**
- `SPACE`: Capture and save  
  - Saving occurs only when **both cameras in the current pair detect chessboard corners**.
- `q` or `ESC`: Exit

---

### 2. Obtain the Chessboard-to-Robot-Base Transform Matrix

```bash
python3 tcp_to_base3.py
```

- Enter the recorded TCP pose `X Y Z Rx Ry Rz` (from when `left_0.png / right_0.png` were taken) into `tcp_pose` (**rotation angles must be (90, 0, XX)**).
- `tcp_pose = [x, y, z, 90, 0, rz]` is the TCP pose in the **Base** frame (read from TMflow Point Manager).
- `tcp_chess_pose = [-106.5, 71, 95, 0, 180, 180]` is the Chessboard pose in the **TCP** frame. `x y z` must be computed based on the fixture offset (mm).
- When mounted, the chessboard’s black squares must face upward.
- After confirming the 3D visualization is correct, copy the `T_base_chess_Rz180` matrix and paste it into the main program to replace `T_offset_12 / T_offset_34`.

---

### 3. Run the Main Program

```bash
python3 fourcamsafe.areav13.2floor.py
```

Adjust parameters according to the table below.

## Parameter Tuning for `fourcamsafe.areav13.2floor`

| Name | Description |
| :--- | :--- |
| `CALIB_ROOT` | Parent directory containing calibration photo folders for pair12 / pair34 |
| `MODEL_PATH` | Absolute path to the skeleton model weights |
| `REID_WEIGHTS` | Absolute path to the OSNet weights |
| `CAM1_IDX` | Camera index |
| `USE_PAIR12/34` | Enable/disable the pair |
| `APPLY_ALIGN_PAIR12/34` | Toggle skeleton alignment offset |
| `AABB_SCALE` | Dynamic emergency stop zone size (unitless) |
| `SLOW_BOX_HALF_EXTENT_M` | Half-extent (radius) of the fixed blue slow-down zone (m) |
| `FLOOR_TEXTURE_PATH` | Floor texture path for the 3D physics simulation environment |
| `T_offset_12/34` | Transform matrix computed by `tcp_to_base3.py` |
| `self.tm5_id = p.loadURDF` | Loads `tm_description/urdf/tm5-900.urdf` |

URDF folder download link: https://github.com/TechmanRobotInc/tmr_ros2/tree/humble/tm_description

---

## Parameter Tuning for `safe.areav12.4floor`

| Name | Description |
| :--- | :--- |
| `CALIB_ROOT` | Parent directory containing calibration photo folders for pair12 / pair34 |
| `MODEL_PATH` | Absolute path to the skeleton model weights |
| `CAM_LINUX_INDEX` | Camera index |
| `USE_PAIR12/34` | Enable/disable the pair |
| `BIAS_XY_M` | Skeleton alignment offset for pair12 (apply pair12 first, then pair34) |
| `APPLY_ALIGN_PAIR_34` | Toggle skeleton alignment offset for pair34 |
| `AABB_SCALE` | Dynamic emergency stop zone size (unitless) |
| `SLOW_BOX_HALF_EXTENT_M` | Half-extent (radius) of the fixed blue slow-down zone (m) |
| `FLOOR_TEXTURE_PATH` | Floor texture path for the 3D physics simulation environment |
| `T_offset_12/34` | Transform matrix computed by `tcp_to_base3.py` |
| `self.tm5_id = p.loadURDF` | Loads `tm_description/urdf/tm5-900.urdf` |

URDF folder download link: https://github.com/TechmanRobotInc/tmr_ros2/tree/humble/tm_description







**人機協作安全光柵**
繞式佈署多台相機，結合多人人體三維姿態追蹤技術，掌握各個關節與機械手臂之間的空間距離與運動關係，持續監測是否有人體骨架進入人機協作區域。

## 目錄結構與檔案說明

| 類別 | 檔案名稱 | 說明 |
| :--- | :--- | :--- |
| **主程式** | `fourcamsafe.areav13.2floor.py` | **人機協作安全光柵主程式** (有使用OSnet 效能負擔較重) |
| **備用主程式** | `safe.areav12.4floor.py` | **人機協作安全光柵主程式** (效能負擔輕 鬼影較多)|
| **四相機拍攝程式** | `4capture.py` | 用4相機拍攝校正照片與手眼位置照片拍攝 |
| **雙相機拍攝程式** | `2capture.py` | 用2相機拍攝校正照片與手眼位置照片拍攝(如果你需要) |
| **手眼座標轉換** | `tcp_to_base3.py` | 將機械手臂姿態輸入即可得到棋盤格到手臂基座的轉換矩陣 輸入至主程式中 |
| **人體骨架權重檔** | `human.pt` | 用於2D人體骨架推論 程式路徑導入 |
| **OSnet權重檔** | `osnet_x0_25_msmt17.pt` | 用於多人人體匹配 程式路徑導入 |
| **3D模擬環境地板貼圖** | `images.png` | 用於替換地板材質視覺效果較佳 |
| **URDF資料夾** | `tm_description` | 用於3D物理模擬環境之機械手臂模型 |
---

## 校正流程與工具

若相機位置或視角有變動，請依序執行以下步驟重新校正

1. **蒐集校正照片**：
```bash
cd (至4capture.py資料夾位置)
```
```bash
python3 4capture.py
```
此腳本用於 四台相機同步預覽，並以「兩兩成對」方式擷取立體標定用的棋盤格影像。

* 存檔為 原始 PNG（不含疊圖），檔名為 left_{N}.png / right_{N}.png，儲存路徑可依環境更改。
* left_0.png / right_0.png為棋盤格鎖固至手臂末端上照片(需紀錄當下TCP姿態XYZRxRyRz，**旋轉角只能為(90,0,XX)**)。
* 程式會掃描資料夾內既有檔名，自動接續編號，避免覆蓋
* CAM_SOURCES: 四台相機來源會依照USB插入順序編號（為整數 index 或 /dev/videoX）範例：[0, 2, 4, 6]
* TARGET_SIZE: 擷取解析度，不可更改（存檔也使用此解析度）範例：(1920, 1080)
### 鍵盤操作

- `1`：切換到 **pair12（Cam1–Cam2）**
- `3`：切換到 **pair34（Cam3–Cam4）**
- `SPACE`：拍照存檔  
  - 只有在「目前模式的兩台相機都偵測到棋盤格角點」才會存檔  
- `q` 或 `ESC`：離開程式



2. **獲取棋盤格到手臂基座的轉換矩陣**：
```bash
python3 tcp_to_base3.py
```
* 將拍攝left_0.png / right_0.png當下TCP姿態XYZRxRyRz輸入至程式tcp_pose(**旋轉角只能為(90,0,XX)**)。
* tcp_pose = [x, y, z, 90, 0, rz] Base 座標系下的 TCP 位姿(從TMflow中點位管理員看)。
* tcp_chess_pose = [-106.5, 71, 95, 0, 180, 180] TCP 座標系下的 Chessboard 位姿，x y z需根據治具偏移量計算(mm)。
* 鎖固時棋盤格黑色方格需在上。
* 確認3D圖示沒問題後複製T_base_chess_Rz180矩陣貼到主程式T_offset_12/T_offset_34中替換。


3. **主程式**：
```bash
python3 fourcamsafe.areav13.2floor.py
```
參數調整依以下表格介紹


## fourcamsafe.areav13.2floor主程式參數調整

| 名稱 | 說明 |
| :--- | :--- |
| `CALIB_ROOT` | pair12/pair34 校正照片資料夾之母資料夾位置 |
| `MODEL_PATH` | 骨架權重檔絕對路徑 |
| `REID_WEIGHTS` | Osnet權重檔絕對路徑 |
| `CAM1_IDX` | 相機編號 |
| `USE_PAIR12/34` | 開啟關閉配對 |
| `APPLY_ALIGN_PAIR12/34` | 可開關 骨架對齊偏移量 |
| `AABB_SCALE` | 動態緊急停止區大小設定(無單位) |
| `SLOW_BOX_HALF_EXTENT_M` | 藍色固定減速區半徑設定(m) |
| `FLOOR_TEXTURE_PATH` | 3D物理模擬環境之地板圖片導入路徑 |
| `T_offset_12/34` | 由tcp_to_base3.py計算之轉換矩陣 |
| `self.tm5_id = p.loadURDF` | 導入tm_description/urdf/tm5-900.urdf |
URDF資料夾下載路徑：https://github.com/TechmanRobotInc/tmr_ros2/tree/humble/tm_description



## safe.areav12.4floor主程式參數調整

| 名稱 | 說明 |
| :--- | :--- |
| `CALIB_ROOT` | pair12/pair34 校正照片資料夾之母資料夾位置 |
| `MODEL_PATH` | 骨架權重檔絕對路徑 |
| `CAM_LINUX_INDEX` | 相機編號 |
| `USE_PAIR12/34` | 開啟關閉配對 |
| `BIAS_XY_M` | 12骨架對齊偏移量 先12再34 |
| `APPLY_ALIGN_PAIR_34` | 可開關 34骨架對齊偏移量 |
| `AABB_SCALE` | 動態緊急停止區大小設定(無單位) |
| `SLOW_BOX_HALF_EXTENT_M` | 藍色固定減速區半徑設定(m) |
| `FLOOR_TEXTURE_PATH` | 3D物理模擬環境之地板圖片導入路徑 |
| `T_offset_12/34` | 由tcp_to_base3.py計算之轉換矩陣 |
| `self.tm5_id = p.loadURDF` | 導入tm_description/urdf/tm5-900.urdf |
URDF資料夾下載路徑：https://github.com/TechmanRobotInc/tmr_ros2/tree/humble/tm_description
