import math

import cv2
import numpy as np

from model.salient_model import SalientDetection
import torch
from dataclasses import dataclass


@dataclass
class FeedData:
    img_w: int
    img_h: int
    data: np.ndarray

    def as_dict(self):
        return {
            "img_w": self.img_w,
            "img_h": self.img_h,
            "data": self.data
        }


def order_points(pts):
    """按顺时针顺序排列点（左上、右上、右下、左下）"""
    rect = np.zeros((4, 2), dtype="float32")

    # 左上点有最小和，右下点有最大和
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 右上点有最小差，左下点有最大差
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def add_small_rotation(M, image_width, image_height, angle_degrees=-0.5):
    """
    给变换矩阵添加小角度旋转

    参数:
        M: 原始变换矩阵 (3x3)
        angle_degrees: 旋转角度(负数表示逆时针), 默认-0.5度

    返回:
        添加旋转后的变换矩阵
    """
    # 将角度转换为弧度
    angle_radians = angle_degrees * np.pi / 180.0

    # 创建旋转矩阵
    cos_theta = math.cos(angle_radians)
    sin_theta = math.sin(angle_radians)

    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # 旋转通常应围绕图像中心进行
    height, width = image_height, image_width  # 需要替换为实际图像尺寸
    center_x, center_y = width / 2, height / 2

    # 创建以图像中心为原点的旋转矩阵
    T_to_center = np.array([
        [1, 0, -center_x],
        [0, 1, -center_y],
        [0, 0, 1]
    ], dtype=np.float32)

    T_from_center = np.array([
        [1, 0, center_x],
        [0, 1, center_y],
        [0, 0, 1]
    ], dtype=np.float32)

    # 完整变换: 先平移到原点，然后旋转，再平移回来
    R_around_center = T_from_center @ rotation_matrix @ T_to_center

    # 将旋转与原始变换组合
    # 注意：矩阵乘法顺序为后应用的变换在左边
    result_matrix = R_around_center @ M

    return result_matrix


def is_circular(contour, threshold=0.9):
    """
    判断轮廓是否为圆形

    参数:
        contour: 轮廓点集
        threshold: 圆度阈值，范围[0,1]，越接近1表示越接近圆形

    返回:
        布尔值，表示是否为圆形
    """
    # 计算轮廓的面积和周长
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # 理想圆形的面积和周长关系: area = π * r²，perimeter = 2π * r
    # 从周长推算理想圆的半径和面积
    radius = perimeter / (2 * np.pi)
    ideal_area = np.pi * radius * radius

    # 计算圆度(圆形度量)
    if ideal_area > 0:
        circularity = area / ideal_area
    else:
        return False

    return circularity > threshold


def main():
    model = SalientDetection("pytorch_model.pt")
    img = cv2.imread("data/_DSC0498.JPG")
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]
    data = model.preprocess(img)
    predicted = model.inference(data)
    feed_data = FeedData(img_w, img_h, predicted)
    mask = model.postprocess(feed_data.as_dict())
    # mask = cv2.merge([mask, mask, mask])
    # img = cv2.bitwise_and(img, mask)
    # gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # blured = cv2.GaussianBlur(gray_img, [5, 5], 0)
    # edges = cv2.Canny(mask, 5, 125)

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(img, [largest_contour], -1, (0, 255, 255))

    if is_circular(largest_contour, threshold=0.85):
        # 对圆形使用最小外接矩形(可能旋转)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="float32")

        # 重新排列点(以顺时针顺序)
        rect = order_points(box)

        # 用于调试显示
        for point in box:
            point_int = tuple(map(int, point))
            img = cv2.circle(img, point_int, 10, (255, 0, 0), 2)
    else:

        # 将轮廓近似为四边形
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # 调整epsilon以获取四个点
        while len(approx) > 4:
            epsilon *= 1.05
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # print(approx)
        for i in range(4):
            point = approx[i][0]
            print(point)
            img = cv2.circle(img, point, 10, (0, 0, 255), 1)
        # cv2.imshow("points", img)
        # cv2.waitKey(0)

        # 如果无法得到恰好4个点，使用边界矩形
        if len(approx) != 4:
            x, y, w, h = cv2.boundingRect(largest_contour)
            approx = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32).reshape(4, 1, 2)

        # 重新排列点
        pts = approx.reshape(4, 2)
        rect = order_points(pts)

    # 计算目标正方形尺寸
    side_length = max(
        np.linalg.norm(rect[1] - rect[0]),  # 上边
        np.linalg.norm(rect[2] - rect[1]),  # 右边
        np.linalg.norm(rect[3] - rect[2]),  # 下边
        np.linalg.norm(rect[0] - rect[3])  # 左边
    )
    side_length = int(side_length)

    # 定义目标正方形坐标
    dst = np.array([
        [0, 0],  # 左上
        [side_length - 1, 0],  # 右上
        [side_length - 1, side_length - 1],  # 右下
        [0, side_length - 1]  # 左下
    ], dtype="float32")

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)

    # 应用透视变换
    warped = cv2.warpPerspective(img, M, (side_length, side_length))

    # 自定义旋转
    # image_h, image_w = warped.shape[:2]
    # center = image_w / 2, image_h / 2
    # rotation_matrix = cv2.getRotationMatrix2D(center, -0.4, 1.0)
    # warped = cv2.warpAffine(warped, rotation_matrix, (image_w, image_h))

    cv2.imshow("image", warped[..., ::-1])
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
