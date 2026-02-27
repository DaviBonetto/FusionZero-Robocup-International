import argparse
import sys
from pathlib import Path
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
INTL_ROOT = ROOT / "1_international"
if str(INTL_ROOT) not in sys.path:
    sys.path.insert(0, str(INTL_ROOT))

try:
    from behaviours.silver_detection import SilverLineDetector
except Exception as exc:
    SilverLineDetector = None
    print(f"[WARN] SilverLineDetector not available: {exc}")


def draw_label(img, text, x, y, color=(0, 255, 0)):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def perspective_transform(image, width, height):
    top_left = (int(width / 4), int(height / 5))
    top_right = (int(width * 3 / 4), int(height / 5))
    bottom_left = (0, height - 70)
    bottom_right = (width, height - 70)

    src_points = np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)
    dst_points = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, matrix, (width, height))


class LineDetector:
    def __init__(self, width, height):
        self.WIDTH = width
        self.HEIGHT = height
        self.min_black_area = 50
        self.prev_angle = 90
        self.gap_found = 0
        self.green_sign = None
        # Adjustable black threshold (HSV)
        self.black_h_max = 180
        self.black_s_max = 255
        self.black_v_max = 70
        self.black_threshold = ((0, 0, 0), (self.black_h_max, self.black_s_max, self.black_v_max))
        # Adjustable morphology
        self.erode_iter = 3
        self.dilate_iter = 4
        self.erode_ksize = 3
        self.dilate_ksize = 3

    def black_mask(self, image, display_image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.black_threshold = ((0, 0, 0), (self.black_h_max, self.black_s_max, self.black_v_max))
        black_mask = cv2.inRange(hsv_image, self.black_threshold[0], self.black_threshold[1])

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.erode_ksize, self.erode_ksize))
        black_mask = cv2.erode(black_mask, kernel, iterations=self.erode_iter)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.dilate_ksize, self.dilate_ksize))
        black_mask = cv2.dilate(black_mask, kernel, iterations=self.dilate_iter)

        contours, _ = cv2.findContours(black_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_black_area]

        if len(contours) == 1:
            contour = contours[0]
        elif len(contours) == 0:
            contour = None
        else:
            true_prev_angle = self.prev_angle
            angle_difference = []
            tmp_display = None

            if true_prev_angle < 90:
                for contour in contours:
                    angle, _ = self.calculate_angle(contour, tmp_display)
                    angle_difference.append(angle - true_prev_angle)
            elif true_prev_angle > 90:
                for contour in contours:
                    angle, _ = self.calculate_angle(contour, tmp_display)
                    angle_difference.append(true_prev_angle - angle)
            else:
                for contour in contours:
                    angle, _ = self.calculate_angle(contour, tmp_display)
                    angle_difference.append(abs(true_prev_angle - angle))

            min_angle_index = angle_difference.index(min(angle_difference))
            contour = contours[min_angle_index]
            self.calculate_angle(contour, tmp_display)

        if contour is not None and display_image is not None:
            cv2.drawContours(display_image, contours, -1, (255, 0, 0), 2)

        return contour, black_mask

    def calculate_bottom_points(self, contour):
        bottom_edge_points = [(p[0][0], p[0][1]) for p in contour if p[0][1] >= self.HEIGHT - 1]
        bottom_edge_points.sort(key=lambda p: p[0])
        if len(bottom_edge_points) > 2:
            for i in range(1, len(bottom_edge_points)):
                prev_point = bottom_edge_points[i - 1]
                curr_point = bottom_edge_points[i]
                bottom = curr_point[0] - prev_point[0]
                if bottom >= 20:
                    return prev_point, curr_point
        return None

    def calculate_top_contour(self, contour):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)

        sorted_box = sorted(box, key=lambda x: x[1])
        top_left, top_right = sorted_box[:2]
        top_left = tuple(top_left)
        top_right = tuple(top_right)

        return ((top_left[0] + top_right[0]) // 2, (top_left[1] + top_right[1]) // 2)

    def calculate_angle(self, contour, display_image):
        angle = 90

        if contour is not None:
            top_edge_points = [(p[0][0], p[0][1]) for p in contour if p[0][1] <= 5]
            left_edge_points = [(p[0][0], p[0][1]) for p in contour if p[0][0] <= 10]
            right_edge_points = [(p[0][0], p[0][1]) for p in contour if p[0][0] >= self.WIDTH - 10]

            ref_point = None

            if top_edge_points:
                x_avg = int(np.mean([p[0] for p in top_edge_points]))
                ref_point = (x_avg, 0)
            else:
                if self.green_sign is not None:
                    if self.green_sign == "Left":
                        self.prev_angle = 0
                    if self.green_sign == "Right":
                        self.prev_angle = 180

                bottom_points = self.calculate_bottom_points(contour)
                if bottom_points:
                    leftmost_point, rightmost_point = bottom_points

                    if angle < 90:
                        distance = leftmost_point[0] - (self.WIDTH // 2)
                        angle = 90 + int(distance / self.WIDTH * 90)
                        if display_image is not None:
                            cv2.line(display_image, leftmost_point, (leftmost_point[0], 0), (0, 0, 255), 2)
                    elif angle > 90:
                        distance = rightmost_point[0] - (self.WIDTH // 2)
                        angle = 90 + int(distance / self.WIDTH * 90)
                        if display_image is not None:
                            cv2.line(display_image, rightmost_point, (rightmost_point[0], 0), (0, 0, 255), 2)

                elif left_edge_points and right_edge_points:
                    if self.prev_angle < 90:
                        y_avg = int(np.mean([p[1] for p in left_edge_points]))
                        ref_point = (0, y_avg)
                    elif self.prev_angle > 90:
                        y_avg = int(np.mean([p[1] for p in right_edge_points]))
                        ref_point = (self.WIDTH - 1, y_avg)
                    else:
                        left_y_avg = int(np.mean([p[1] for p in left_edge_points]))
                        right_y_avg = int(np.mean([p[1] for p in right_edge_points]))
                        if left_y_avg < right_y_avg:
                            ref_point = (0, left_y_avg)
                        elif left_y_avg > right_y_avg:
                            ref_point = (self.WIDTH - 1, right_y_avg)
                else:
                    top_ref_point = self.calculate_top_contour(contour)
                    if top_ref_point[1] < self.HEIGHT / 2:
                        ref_point = top_ref_point
                    elif left_edge_points:
                        y_avg = int(np.mean([p[1] for p in left_edge_points]))
                        ref_point = (0, y_avg)
                    elif right_edge_points:
                        y_avg = int(np.mean([p[1] for p in right_edge_points]))
                        ref_point = (self.WIDTH, y_avg)

            if ref_point:
                bottom_center = (self.WIDTH // 2, self.HEIGHT)
                dx = bottom_center[0] - ref_point[0]
                dy = bottom_center[1] - ref_point[1]

                angle_radians = np.arctan2(dy, dx)
                angle = int(np.degrees(angle_radians))
                self.prev_angle = angle

                if display_image is not None:
                    cv2.line(display_image, ref_point, bottom_center, (0, 0, 255), 2)
                self.gap_found = 0
            else:
                self.gap_found += 1
        else:
            self.gap_found += 1

        return angle, self.gap_found


class BallDetector:
    def __init__(self, width, height, display=True):
        self.DISPLAY = display
        self.DEBUG_LIVE = False
        self.DEBUG_DEAD = False
        self.TIMING_LIVE = False
        self.TIMING_DEAD = False

        self.CROP_SIZE = 200
        self.IMG_CENTRE_X = width / 2

        # Silver ball (shape-based) Hough params
        self.SILVER_BLUR = 7
        self.SILVER_HOUGH_DP = 1.2
        self.SILVER_HOUGH_MIN_DISTANCE = 60
        self.SILVER_HOUGH_PARAM1 = 120
        self.SILVER_HOUGH_PARAM2 = 30
        self.SILVER_HOUGH_MIN_RADIUS = 8
        self.SILVER_HOUGH_MAX_RADIUS = 120

        self.DEAD_GREEN_KERNAL = np.ones((7, 7), np.uint8)
        self.DEAD_WHITE_KERNAL = np.ones((9, 9), np.uint8)
        self.DEAD_BLACK_KERNAL = np.ones((25, 25), np.uint8)
        self.DEAD_WHITE_THRESHOLD = (160, 160, 160)
        self.DEAD_BLACK_THRESHOLD = (60, 60, 60)
        self.DEAD_MIN_BLACK_AREA = 300
        self.DEAD_MIN_Y = 40
        self.DEAD_RADIUS_Y_MIN = -50
        self.HOUGH_DP = 1
        self.HOUGH_MIN_DISTANCE = 200
        self.HOUGH_PARAMETER_1 = 50
        self.HOUGH_PARAMETER_2 = 30
        self.HOUGH_MIN_RADIUS = 5
        self.HOUGH_MAX_RADIUS = 150
        self.last_live_circle = None
        self.last_dead_circle = None

    def live(self, image, display_image, last_x, draw=True):
        """
        Silver ball detection by shape (Hough circles).
        """
        if image is None or image.size == 0:
            return None

        if last_x is not None:
            x_lower = max(0, last_x - self.CROP_SIZE)
            x_upper = min(image.shape[1], last_x + self.CROP_SIZE)
            working_image = image[:, x_lower:x_upper]
        else:
            working_image = image.copy()
            x_lower = 0

        gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        k = int(self.SILVER_BLUR)
        if k % 2 == 0:
            k += 1
        k = max(3, k)
        gray = cv2.GaussianBlur(gray, (k, k), 0)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            self.SILVER_HOUGH_DP,
            self.SILVER_HOUGH_MIN_DISTANCE,
            param1=self.SILVER_HOUGH_PARAM1,
            param2=self.SILVER_HOUGH_PARAM2,
            minRadius=self.SILVER_HOUGH_MIN_RADIUS,
            maxRadius=self.SILVER_HOUGH_MAX_RADIUS,
        )

        if circles is None:
            return None

        circles = np.round(circles[0, :]).astype("int")
        if last_x is not None:
            crop_center = working_image.shape[1] / 2
            best_circle = min(circles, key=lambda c: abs(c[0] - crop_center))
        else:
            best_circle = max(circles, key=lambda c: c[2])

        x, y, r = best_circle
        centre_x = x + x_lower
        self.last_live_circle = (centre_x, y, r)

        if self.DISPLAY and draw and display_image is not None:
            cv2.circle(display_image, (centre_x, y), r, (0, 255, 255), 2)
            cv2.circle(display_image, (centre_x, y), 2, (0, 255, 255), 2)

        return int(centre_x)

    def dead(self, image, display_image, last_x):
        if image is None or image.size == 0:
            return None

        if last_x is not None:
            x_lower = max(0, last_x - self.CROP_SIZE)
            x_upper = min(image.shape[1], last_x + self.CROP_SIZE)
            working_image = image[:, x_lower:x_upper]
        else:
            working_image = image.copy()
            x_lower = 0

        white = cv2.inRange(working_image, self.DEAD_WHITE_THRESHOLD, (255, 255, 255))
        white = cv2.erode(white, self.DEAD_WHITE_KERNAL, iterations=1)
        white = cv2.dilate(white, self.DEAD_WHITE_KERNAL, iterations=1)
        working_image[white > 0] = [160, 160, 160]

        black_mask = cv2.inRange(working_image, (0, 0, 0), self.DEAD_BLACK_THRESHOLD)

        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(black_mask)
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area >= self.DEAD_MIN_BLACK_AREA:
                cv2.fillPoly(filtered_mask, [contour], 255)
        black_mask = filtered_mask
        black_mask = cv2.dilate(black_mask, self.DEAD_BLACK_KERNAL, iterations=1)

        grey_image = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        grey_image = cv2.bitwise_and(grey_image, black_mask)

        circles = cv2.HoughCircles(
            grey_image,
            cv2.HOUGH_GRADIENT,
            self.HOUGH_DP,
            self.HOUGH_MIN_DISTANCE,
            param1=self.HOUGH_PARAMETER_1,
            param2=self.HOUGH_PARAMETER_2,
            minRadius=self.HOUGH_MIN_RADIUS,
            maxRadius=self.HOUGH_MAX_RADIUS,
        )
        if circles is None:
            return None

        circles = np.round(circles[0, :]).astype("int")
        valid = []
        for (x, y, r) in circles:
            x = max(0, min(x, working_image.shape[1] - 1))
            y = max(0, min(y, working_image.shape[0] - 1))
            if y > self.DEAD_MIN_Y and y - r > self.DEAD_RADIUS_Y_MIN:
                valid.append((x, y, r))

        if len(valid) == 0:
            return None

        if last_x is not None:
            crop_center = working_image.shape[1] / 2
            best_circle = min(valid, key=lambda circle: abs(circle[0] - crop_center))
            centre_x = best_circle[0] + x_lower
        else:
            best_circle = max(valid, key=lambda circle: circle[2])
            centre_x = best_circle[0] + x_lower

        self.last_dead_circle = None
        if self.DISPLAY:
            _, y, r = best_circle
            cv2.circle(display_image, (centre_x, y), r, (0, 255, 0), 2)
            cv2.circle(display_image, (centre_x, y), 1, (0, 255, 0), 2)
            self.last_dead_circle = (centre_x, y, r)

        return int(centre_x)


def open_camera(index, width, height, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def main():
    parser = argparse.ArgumentParser(description="PC vision runner for line + ball detection")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--line-width", type=int, default=320)
    parser.add_argument("--line-height", type=int, default=200)
    parser.add_argument("--no-transform", action="store_true")
    parser.add_argument("--mode", choices=["line", "balls", "all"], default="line")
    parser.add_argument("--model", default=str(ROOT / "5_ai_training_data" / "0_models" / "silver_line" / "silver_detector_pi4_quantized.pt"))
    parser.add_argument("--silver-conf", type=float, default=0.95)
    args = parser.parse_args()

    cap = open_camera(args.camera, args.width, args.height, args.fps)
    if not cap.isOpened():
        print("Failed to open camera.")
        return 1

    line_detector = LineDetector(args.line_width, args.line_height)
    ball_detector = BallDetector(320, 240, display=True)

    silver_detector = None
    if SilverLineDetector is not None:
        try:
            silver_detector = SilverLineDetector(args.model)
        except Exception as exc:
            print(f"[WARN] Could not load silver model: {exc}")

    mode = args.mode
    last_live_x = None
    last_dead_x = None

    print("Controls: [1]=line  [2]=balls  [3]=all  [q/ESC]=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        if mode in ("line", "all"):
            line_frame = cv2.resize(frame, (args.line_width, args.line_height))
            if args.no_transform:
                line_view = line_frame
            else:
                line_view = perspective_transform(line_frame, args.line_width, args.line_height)
            line_display = line_view.copy()

            contour, _ = line_detector.black_mask(line_view, line_display)
            if contour is not None:
                angle, gap = line_detector.calculate_angle(contour, line_display)
                draw_label(line_display, f"ANGLE: {angle}", 5, 20, (0, 255, 0))
                if gap > 0:
                    draw_label(line_display, f"GAP: {gap}", 5, 45, (0, 0, 255))
            else:
                draw_label(line_display, "LINE NOT FOUND", 5, 20, (0, 0, 255))

            if silver_detector is not None:
                try:
                    result = silver_detector.predict(line_view)
                    if isinstance(result, tuple):
                        result = result[0]
                    if result["prediction"] == 1 and result["confidence"] >= args.silver_conf:
                        draw_label(
                            line_display,
                            f"SILVER LINE {result['confidence']:.2f}",
                            5,
                            70,
                            (0, 255, 255),
                        )
                except Exception as exc:
                    draw_label(line_display, f"SILVER ERR: {exc}", 5, 70, (0, 0, 255))

            cv2.imshow("Line", line_display)

        if mode in ("balls", "all"):
            ball_frame = cv2.resize(frame, (320, 240))
            ball_display = ball_frame.copy()

            live_x = ball_detector.live(ball_frame, ball_display, last_live_x)
            dead_x = ball_detector.dead(ball_frame, ball_display, last_dead_x)

            if live_x is not None:
                draw_label(ball_display, "SILVER BALL", 5, 20, (0, 255, 255))
                cv2.line(ball_display, (live_x, 0), (live_x, ball_display.shape[0] - 1), (0, 255, 255), 1)
                last_live_x = live_x

            if dead_x is not None:
                draw_label(ball_display, "BLACK BALL", 5, 45, (0, 0, 0))
                cv2.line(ball_display, (dead_x, 0), (dead_x, ball_display.shape[0] - 1), (0, 0, 0), 2)
                last_dead_x = dead_x

            cv2.imshow("Balls", ball_display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("1"):
            mode = "line"
        elif key == ord("2"):
            mode = "balls"
        elif key == ord("3"):
            mode = "all"

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
