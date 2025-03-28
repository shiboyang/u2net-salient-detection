from pathlib import Path
from typing import List

import cv2
import numpy as np

import utils
from gui import preview_four_point
from model.salient_model import SalientDetection
from utils import Shape
from tqdm import tqdm


class Engin:
    model: SalientDetection
    shape: Shape

    def __init__(
            self,
            model_path: str,
            output_dir: str,
            extension: List[str],
            manual_select: bool = False,
            scale: float = 1.0,
    ):
        self.output_dir = Path(output_dir)
        self.model = SalientDetection(model_path)
        self.manual_select = manual_select
        self.scale = scale
        self.extension = self.process_extension(extension)

    def process_extension(self, extension):
        extensions = []
        for ext in extension:
            ext.replace(".", "")
            extensions.append(ext)
        return extensions

    def model_process(self, img):
        img_h, img_w = img.shape[:2]
        data = self.model.preprocess(img)
        output = self.model.inference(data)
        result = self.model.postprocess(dict(data=output, img_w=img_w, img_h=img_h))
        return result

    def process_single(self, img_path: Path):
        bgr_img = cv2.imread(img_path.as_posix())
        img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        mask = self.model_process(img)
        four_points = self.calculate_four_points(mask, img)
        if self.manual_select:
            four_points = preview_four_point(img, four_points, self.shape, self.scale)
        if four_points is None:
            print(f"esc退出选点模式，跳过图像: {img_path.name}")
            return
        cropped_img = self.trapezoid_correction(four_points, bgr_img)
        cv2.imwrite(self.output_dir.joinpath(img_path.name).as_posix(), cropped_img)

    def trapezoid_correction(self, rect: np.ndarray, img: np.ndarray):
        side_length = max(
            np.linalg.norm(rect[1] - rect[0]),  # top
            np.linalg.norm(rect[2] - rect[1]),  # right
            np.linalg.norm(rect[3] - rect[2]),  # bottom
            np.linalg.norm(rect[0] - rect[3])  # left
        )
        side_length = int(side_length)

        # define rectangle canvas
        dst = np.array([
            [0, 0],  # top left
            [side_length - 1, 0],  # top right
            [side_length - 1, side_length - 1],  # bottom right
            [0, side_length - 1]  # bottom left
        ], dtype="float32")

        # calculate perspective transformation
        M = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
        # apply
        warped = cv2.warpPerspective(img, M, (side_length, side_length))
        return warped

    @staticmethod
    def _is_circular(contour: np.ndarray, threshold=0.8):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        radius = perimeter / (2 * np.pi)
        ideal_area = np.pi * radius * radius
        if ideal_area > 0:
            circularity = area / ideal_area
        else:
            return False
        return circularity > threshold

    def calculate_four_points(self, mask: np.ndarray, img: np.ndarray):
        assert mask.shape[0] == img.shape[0] and mask.shape[1] == img.shape[1]
        thresh, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        if self._is_circular(largest_contour):
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            pts = np.array(box, dtype=np.float32)
            self.shape = Shape.circle
        else:
            # process rectangle
            self.shape = Shape.rectangle
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            # 调整epsilon以获取四个点
            while len(approx) > 4:
                epsilon *= 1.05
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            # for i in range(4):
            #     point = approx[i][0]
            #     print(point)
            #     img = cv2.circle(img, point, 10, (0, 0, 255), 1)
            # cv2.imshow("points", img)
            # cv2.waitKey(0)

            # 如果无法得到恰好4个点，使用边界矩形
            if len(approx) != 4:
                x, y, w, h = cv2.boundingRect(largest_contour)
                approx = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32).reshape(4, 1, 2)
            pts = approx.reshape(4, 2)
        ordered_pts = utils.order_points(pts)
        return ordered_pts

    def run(self, input_path: str):
        img_path = []
        input_path = Path(input_path)
        if input_path.is_dir():
            for ext in self.extension:
                tmp_img_path = input_path.glob(f"*.{ext}")
                img_path.extend(tmp_img_path)

            for img_path in tqdm(img_path):
                self.process_single(img_path)
        elif input_path.is_file():
            suffix = input_path.suffix.replace(".", "")
            if suffix in self.extension:
                self.process_single(input_path)
