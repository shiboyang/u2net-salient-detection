import cv2
import numpy as np
from utils import Shape

window_name: str = "Preview"


def rhombus_to_rectangle(points: np.ndarray):
    rect = np.zeros([4, 2], dtype=np.int32)
    min_x = np.min(points[..., 0])
    min_y = np.min(points[..., 1])
    max_x = np.max(points[..., 0])
    max_y = np.max(points[..., 1])
    rect[0] = min_x, min_y
    rect[1] = max_x, min_y
    rect[2] = max_x, max_y
    rect[3] = min_x, max_y
    return rect


def preview_four_point(img: np.ndarray, points: np.ndarray, shape: Shape = None, scale: float = 1.0):
    clone = img.copy()
    if scale < 1.0:
        clone = cv2.resize(clone, None, fx=scale, fy=scale)
        points *= scale

    points = points.astype(np.int32)
    assert len(points) == 4
    result = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal result
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])

            # Draw circle at clicked point
            cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)

            # Draw point number
            cv2.putText(clone, str(len(points)), (x + 10, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # If we have 4 points, connect them to show polygon
            if len(points) == 4:
                pts = np.array(points, dtype=np.int32)
                if shape is Shape.circle:
                    pts = rhombus_to_rectangle(pts)
                cv2.polylines(clone, [pts], True, (0, 255, 0), 2)
                result = pts.tolist()

            cv2.imshow(window_name, clone)

    # Create window and set mouse callback
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    for i in range(4):
        x, y = points[i]
        clone = cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(clone, f"{i}", (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.polylines(clone, [points], True, (0, 255, 0), 2)

    while True:
        cv2.imshow(window_name, clone)
        key = cv2.waitKey(1) & 0xFF

        # Reset points if 'r' is pressed
        if key == ord('r'):
            clone = img.copy()
            if scale < 1.0:
                clone = cv2.resize(img, None, fx=scale, fy=scale)
            points = []

        # Continue if 'c' is pressed and 4 points selected
        elif key == ord('c') and len(points) == 4:
            break

        # ESC key - exit
        elif key == 27:
            return None

    # cv2.destroyAllWindows()

    if result:
        result = np.array(result, dtype=np.float32)
        result /= scale
        return result.astype(np.int32)
    else:
        return (points.astype(np.float32) / scale).astype(np.int32)
