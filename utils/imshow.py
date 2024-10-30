import cv2
import numpy as np


def imshow(title: str, img: np.ndarray, wait_ms: int = 10):
    """显示图像 Esc键退出"""

    if wait_ms <= 0:
        while True:
            cv2.imshow(title, img)
            if cv2.waitKey(20) == 27:  # Esc
                break
    else:
        cv2.imshow(title, img)
        cv2.waitKey(wait_ms)

def plot_joints_2d(img, kps_2d, color=(255, 0, 0), r=3):
    """在图片上绘制关节点"""

    if isinstance(kps_2d, list):
        kps_2d = np.array(kps_2d, dtype=np.float32)

    if len(kps_2d.shape) == 2:
        kps_2d = kps_2d[None, :, :]

    n, kps_num = kps_2d.shape[:2]
    for i in range(n):
        for j in range(kps_num):
            u, v = kps_2d[i, j, :2]
            cv2.circle(img, (int(u), int(v)), r, color, -1)

def half_img(img: np.ndarray):
    h, w = img.shape[:2]
    return cv2.resize(img, (w // 2, h // 2))
