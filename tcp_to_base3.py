import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (trigger 3D)

"""
Right-handed chessboard transform 
------------------------------------------
• 全程右手座標（det(R)=+1）
• 可選：讓棋盤 +Z 朝向相機（以 Rx(180°) 實現，不用左手 Z-flip）
• 視覺化 Base / TCP / Chessboard 三座標系
"""

# ================== 基本函數 ==================
def euler_zyx_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """依序繞 x, y, z 軸旋轉，回傳 Rz @ Ry @ Rx（與你原本一致）。"""
    R_x = R.from_euler('x', rx_deg, degrees=True).as_matrix()
    R_y = R.from_euler('y', ry_deg, degrees=True).as_matrix()
    R_z = R.from_euler('z', rz_deg, degrees=True).as_matrix()
    return R_z @ R_y @ R_x

def to_homogeneous(Rm: np.ndarray, t_xyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = np.array(t_xyz, dtype=float).reshape(3)
    return T

def tcp_to_homogeneous_matrix(pose6):
    """pose6 = (x, y, z, rx, ry, rz)  (mm, deg)"""
    x, y, z, rx, ry, rz = pose6
    Rm = euler_zyx_matrix(rx, ry, rz)
    return to_homogeneous(Rm, (x, y, z))

def homo_Rz_deg(deg: float) -> np.ndarray:
    """建立繞 Z 軸旋轉 deg 度的 4x4 齊次旋轉矩陣。"""
    Rz = R.from_euler('z', deg, degrees=True).as_matrix()
    Tz = np.eye(4)
    Tz[:3, :3] = Rz
    return Tz

# ================== 核心：右手棋盤變換 ==================
def compute_T_base_chessboard(
    tcp_pose,
    tcp_chess_pose,
    z_toward_camera: bool = True,
):
    """
    回傳：T_base_tcp, T_base_chessboard（皆為右手）
    若 z_toward_camera=True：右乘 Rx(180°)（det=+1），讓棋盤 +Z 指向相機（同時反轉棋盤 Y）。
    """
    T_base_tcp = tcp_to_homogeneous_matrix(tcp_pose)
    T_tcp_chess = tcp_to_homogeneous_matrix(tcp_chess_pose)

    T_base_chess = T_base_tcp @ T_tcp_chess

    if z_toward_camera:
        # 用 Rx(180°) 取代左手 Z-flip：保持右手座標
        R_rx180 = R.from_euler('x', 180, degrees=True).as_matrix()  # 等效於 diag(1,-1,-1)
        T_rx180 = np.eye(4)
        T_rx180[:3, :3] = R_rx180
        T_base_chess = T_base_chess @ T_rx180

    # （可選）數值正交化，避免累積誤差
    U, S, Vt = np.linalg.svd(T_base_chess[:3, :3])
    T_base_chess[:3, :3] = U @ Vt

    return T_base_tcp, T_base_chess

# ================== 視覺化 ==================
def plot_frame(ax, T: np.ndarray, label: str, axis_length: float = 100.0,
               colors=('r', 'g', 'b')):
    o = T[:3, 3]
    x_axis = o + T[:3, 0] * axis_length
    y_axis = o + T[:3, 1] * axis_length
    z_axis = o + T[:3, 2] * axis_length
    ax.quiver(*o, *(x_axis - o), color=colors[0], arrow_length_ratio=0.1)
    ax.quiver(*o, *(y_axis - o), color=colors[1], arrow_length_ratio=0.1)
    ax.quiver(*o, *(z_axis - o), color=colors[2], arrow_length_ratio=0.1)
    ax.text(*o, label, fontsize=12)

def plot_3d_environment(T_base_tcp, T_base_chessboard, axis_length=100.0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Base / TCP / Chessboard
    plot_frame(ax, np.eye(4), 'Base', axis_length)
    plot_frame(ax, T_base_tcp, 'TCP', axis_length)
    plot_frame(ax, T_base_chessboard, 'Chessboard', axis_length)

    ax.set_xlim(-100, 800)
    ax.set_ylim(-100, 800)
    ax.set_zlim(-100, 800)
    ax.set_xticks(np.arange(-100, 801, 100))
    ax.set_yticks(np.arange(-100, 801, 100))
    ax.set_zticks(np.arange(-100, 801, 100))
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Right-handed Base/TCP/Chessboard frames')
    plt.tight_layout()
    plt.show()

# ================== 驗證小工具 ==================
def detR(T: np.ndarray) -> float:
    return float(np.linalg.det(T[:3, :3]))

def print_T(name: str, T: np.ndarray):
    np.set_printoptions(suppress=True, precision=3)
    print(f"\n{name}:")
    print(T)
    print(f"det(R) = {detR(T):.6f}")

# ================== 主程式 ==================
if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=2)

    # 你的輸入（可改）
    # tcp_pose = [500, -100, 700, 90, 0, 90]
    # tcp_pose =[500, -100, 700, 90, 0, 45]
    # tcp_pose =[100, 100, 1000, 90, 0, -120]
    # tcp_pose =[100, 0, 800, 90, 0, -180]
    tcp_pose =[450, 0, 700, 90, 0, 120]
    tcp_chess_pose = [-106.5, 71, 95, 0, 180, 180]

    # 計算（右手），並讓棋盤 +Z 朝相機（不使用左手 flip）
    T_base_tcp, T_base_chess = compute_T_base_chessboard(
        tcp_pose, tcp_chess_pose, z_toward_camera=True
    )

    # === 新增輸出：T_base_chess 右乘「Z 軸 180°」的結果 ===
    T_z180 = homo_Rz_deg(180.0)
    T_base_chess_Rz180 = T_base_chess @ T_z180

    # 印出矩陣與 det(R)
    formatter = {'float_kind': lambda x: format(x, ',.4f')}
    print_T('T_base_tcp', T_base_tcp)

    print('\nT_base_chess')
    print(np.array2string(T_base_chess, formatter=formatter, separator=', '))
    print(f"det(R) = {detR(T_base_chess):.6f}")

    print('\nT_base_chess_Rz180  (T_base_chess 右乘 Rz(180°))')
    print(np.array2string(T_base_chess_Rz180, formatter=formatter, separator=', '))
    print(f"det(R) = {detR(T_base_chess_Rz180):.6f}")

    # 視覺化（維持原本，只畫未 Rz 的棋盤；如要一起畫，呼叫 plot_frame 再加一個）
    plot_3d_environment(T_base_tcp, T_base_chess, axis_length=100.0)
