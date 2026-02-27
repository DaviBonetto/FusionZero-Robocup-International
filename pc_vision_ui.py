import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from pc_vision_runner import LineDetector, BallDetector, perspective_transform, SilverLineDetector


def open_camera(index, width, height, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


class VisionUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("OBR Integrated System - PC Vision")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.geometry("1200x720")
        self.root.minsize(1000, 600)

        self.video_label = ttk.Label(self.root)
        self.video_label.configure(anchor="nw")
        self.video_label.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")

        self.ctrl_canvas = tk.Canvas(self.root, width=360, highlightthickness=0)
        self.ctrl_scroll = ttk.Scrollbar(self.root, orient="vertical", command=self.ctrl_canvas.yview)
        self.ctrl_canvas.configure(yscrollcommand=self.ctrl_scroll.set)
        self.ctrl_canvas.grid(row=0, column=1, sticky="ns", padx=(0, 8), pady=8)
        self.ctrl_scroll.grid(row=0, column=2, sticky="ns", padx=(0, 8), pady=8)

        self.ctrl = ttk.Frame(self.ctrl_canvas)
        self.ctrl_canvas.create_window((0, 0), window=self.ctrl, anchor="nw")

        def _on_ctrl_configure(_event):
            self.ctrl_canvas.configure(scrollregion=self.ctrl_canvas.bbox("all"))

        self.ctrl.bind("<Configure>", _on_ctrl_configure)

        def _on_mousewheel(event):
            if event.delta:
                self.ctrl_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.ctrl_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)

        self.default_pairs = []

        self._build_controls()

        self.cap_line = None
        self.cap_ball = None
        self._open_cameras()

        self.line_detector = LineDetector(self.line_width.get(), self.line_height.get())
        self.ball_detector = BallDetector(320, 240, display=True)

        self.silver_detector = None
        if SilverLineDetector is not None:
            model_path = Path(self.model_path.get()).resolve()
            if model_path.exists():
                try:
                    self.silver_detector = SilverLineDetector(str(model_path))
                except Exception as exc:
                    self.status_var.set(f"Silver model error: {exc}")
            else:
                self.status_var.set(f"Silver model not found: {model_path}")

        self.last_live_x = None
        self.last_dead_x = None
        self.last_fps_time = time.time()
        self.frame_count = 0

        self.running = True
        self.update_frame()

    def _build_controls(self):
        # Camera
        cam_frame = ttk.LabelFrame(self.ctrl, text="Camera")
        cam_frame.pack(fill="x", pady=4)

        self.camera_index = tk.IntVar(value=0)
        self.camera_width = tk.IntVar(value=640)
        self.camera_height = tk.IntVar(value=480)
        self.camera_fps = tk.IntVar(value=30)

        cam_form = ttk.Frame(cam_frame)
        cam_form.pack(fill="x", pady=2)
        cam_form.grid_columnconfigure(0, weight=1)
        cam_form.grid_columnconfigure(1, weight=0)

        row = 0
        row = self._add_camera_row(cam_form, row, "Index", self.camera_index, "Camera index (PC built-in is usually 0)")
        row = self._add_camera_row(cam_form, row, "Width", self.camera_width, "Capture width in pixels")
        row = self._add_camera_row(cam_form, row, "Height", self.camera_height, "Capture height in pixels")
        row = self._add_camera_row(cam_form, row, "FPS", self.camera_fps, "Target FPS (best effort)")

        tk.Button(cam_frame, text="Reconnect Camera", command=self._open_cameras).pack(fill="x", pady=4)

        # Line detection
        line_frame = ttk.LabelFrame(self.ctrl, text="Line (Preta)")
        line_frame.pack(fill="x", pady=4)

        self.line_width = tk.IntVar(value=320)
        self.line_height = tk.IntVar(value=200)
        self._add_entry(line_frame, "Line Width", self.line_width, "Line processing width")
        self._add_entry(line_frame, "Line Height", self.line_height, "Line processing height")

        self.line_use_transform = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            line_frame,
            text="Perspective transform (corrige angulo)",
            variable=self.line_use_transform,
        ).pack(anchor="w")

        self.line_v_max = self._add_slider(
            line_frame,
            "Black V max",
            0,
            255,
            70,
            1,
            "Max brilho para considerar preto (menor = mais escuro)",
            int_var=True,
        )
        self.line_s_max = self._add_slider(
            line_frame,
            "Black S max",
            0,
            255,
            255,
            1,
            "Max saturacao para considerar preto",
            int_var=True,
        )
        self.line_min_area = self._add_slider(
            line_frame,
            "Min area",
            0,
            2000,
            50,
            1,
            "Area minima do contorno da linha",
            int_var=True,
        )
        self.line_erode = self._add_slider(
            line_frame,
            "Erode iter",
            0,
            10,
            3,
            1,
            "Remove ruido (mais alto = mais agressivo)",
            int_var=True,
        )
        self.line_dilate = self._add_slider(
            line_frame,
            "Dilate iter",
            0,
            10,
            4,
            1,
            "Preenche falhas (mais alto = mais agressivo)",
            int_var=True,
        )

        # Silver line AI
        silver_frame = ttk.LabelFrame(self.ctrl, text="Silver Line AI")
        silver_frame.pack(fill="x", pady=4)

        default_model = str(
            Path(__file__).resolve().parent
            / "5_ai_training_data"
            / "0_models"
            / "silver_line"
            / "silver_detector_pi4_quantized.pt"
        )
        self.model_path = tk.StringVar(value=default_model)
        self._add_entry(silver_frame, "Model", self.model_path, "Path do modelo .pt")

        self.silver_conf = self._add_slider(
            silver_frame,
            "Confidence",
            0.0,
            1.0,
            0.95,
            0.01,
            "Confianca minima para mostrar SILVER LINE",
            int_var=False,
        )
        self.silver_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(silver_frame, text="Enable Silver AI", variable=self.silver_enabled).pack(anchor="w")

        # Balls (silver) - shape based
        live_frame = ttk.LabelFrame(self.ctrl, text="Ball - Silver (Shape)")
        live_frame.pack(fill="x", pady=4)
        self.silver_blur = self._add_slider(
            live_frame,
            "Blur (odd)",
            3,
            21,
            7,
            2,
            "Desfoque para suavizar ruido (impar)",
            int_var=True,
        )
        self.silver_dp = self._add_slider(
            live_frame,
            "Hough dp",
            1.0,
            3.0,
            1.2,
            0.1,
            "Resolucao acumulador (maior = mais rapido, menos sensivel)",
            int_var=False,
        )
        self.silver_min_dist = self._add_slider(
            live_frame,
            "Min dist",
            10,
            300,
            60,
            1,
            "Distancia minima entre circulos",
            int_var=True,
        )
        self.silver_p1 = self._add_slider(
            live_frame,
            "Canny p1",
            10,
            300,
            120,
            1,
            "Borda: menor = detecta mais",
            int_var=True,
        )
        self.silver_p2 = self._add_slider(
            live_frame,
            "Hough p2",
            5,
            100,
            30,
            1,
            "Circulo: menor = aceita mais",
            int_var=True,
        )
        self.silver_min_r = self._add_slider(
            live_frame,
            "Min radius",
            1,
            200,
            8,
            1,
            "Raio minimo do circulo",
            int_var=True,
        )
        self.silver_max_r = self._add_slider(
            live_frame,
            "Max radius",
            10,
            300,
            120,
            1,
            "Raio maximo do circulo",
            int_var=True,
        )

        # Balls (black)
        dead_frame = ttk.LabelFrame(self.ctrl, text="Ball - Black (Dead)")
        dead_frame.pack(fill="x", pady=4)
        self.silver_black_overlap = self._add_slider(
            dead_frame,
            "Overlap px",
            0,
            200,
            40,
            1,
            "Se prata e preta estiverem muito perto, ignora prata",
            int_var=True,
        )
        self.dead_black_thresh = self._add_slider(
            dead_frame,
            "Black V max",
            0,
            120,
            60,
            1,
            "Max brilho para considerar preto",
            int_var=True,
        )
        self.hough_p1 = self._add_slider(
            dead_frame,
            "Hough param1",
            1,
            200,
            50,
            1,
            "Borda: menor = detecta mais",
            int_var=True,
        )
        self.hough_p2 = self._add_slider(
            dead_frame,
            "Hough param2",
            1,
            100,
            30,
            1,
            "Circulo: menor = aceita mais",
            int_var=True,
        )
        self.hough_min_r = self._add_slider(
            dead_frame,
            "Min radius",
            1,
            200,
            5,
            1,
            "Raio minimo do circulo",
            int_var=True,
        )
        self.hough_max_r = self._add_slider(
            dead_frame,
            "Max radius",
            1,
            300,
            150,
            1,
            "Raio maximo do circulo",
            int_var=True,
        )

        # Status
        status_frame = ttk.LabelFrame(self.ctrl, text="Status")
        status_frame.pack(fill="x", pady=4)
        self.status_var = tk.StringVar(value="Ready")
        self.fps_var = tk.StringVar(value="FPS: 0")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.fps_var).pack(anchor="w")

        ttk.Button(self.ctrl, text="Reset Defaults", command=self.reset_defaults).pack(fill="x", pady=6)

    def _add_entry(self, parent, label, var, desc):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        ttk.Label(frame, text=label).pack(side="left")
        ttk.Entry(frame, textvariable=var, width=12).pack(side="right")
        ttk.Label(parent, text=desc, foreground="#555").pack(anchor="w")
        self.default_pairs.append((var, var.get()))

    def _add_camera_row(self, parent, row, label, var, desc):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        entry = tk.Entry(parent, textvariable=var, width=8)
        entry.grid(row=row, column=1, sticky="e", padx=(8, 0))
        ttk.Label(parent, text=desc, foreground="#555").grid(row=row + 1, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self.default_pairs.append((var, var.get()))
        return row + 2

    def _add_slider(self, parent, label, from_, to, init, resolution, desc, int_var=True):
        ttk.Label(parent, text=label).pack(anchor="w")
        var = tk.IntVar(value=init) if int_var else tk.DoubleVar(value=init)
        scale = tk.Scale(
            parent,
            from_=from_,
            to=to,
            orient="horizontal",
            resolution=resolution,
            showvalue=True,
            variable=var,
            length=220,
        )
        scale.pack(fill="x")
        ttk.Label(parent, text=desc, foreground="#555").pack(anchor="w")
        self.default_pairs.append((var, init))
        return var

    def reset_defaults(self):
        for var, value in self.default_pairs:
            try:
                var.set(value)
            except Exception:
                pass

    def _open_cameras(self):
        for cap in (self.cap_line, self.cap_ball):
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

        index = int(self.camera_index.get())
        self.cap_line = open_camera(
            index,
            int(self.camera_width.get()),
            int(self.camera_height.get()),
            int(self.camera_fps.get()),
        )
        self.cap_ball = open_camera(
            index,
            int(self.camera_width.get()),
            int(self.camera_height.get()),
            int(self.camera_fps.get()),
        )

        line_ok = self.cap_line.isOpened() if self.cap_line is not None else False
        ball_ok = self.cap_ball.isOpened() if self.cap_ball is not None else False

        if line_ok and ball_ok:
            self.status_var.set("Cameras ready (line + ball).")
        elif line_ok and not ball_ok:
            self.status_var.set("Ball camera not opened. Using line camera only.")
        elif ball_ok and not line_ok:
            self.status_var.set("Line camera not opened. Using ball camera only.")
        else:
            self.status_var.set("No camera opened. Close other apps and reconnect.")

    def update_frame(self):
        if not self.running:
            return

        ok_line, frame_line = (False, None)
        ok_ball, frame_ball = (False, None)

        if self.cap_line is not None and self.cap_line.isOpened():
            ok_line, frame_line = self.cap_line.read()
        if self.cap_ball is not None and self.cap_ball.isOpened():
            ok_ball, frame_ball = self.cap_ball.read()

        if not ok_line and not ok_ball:
            self.status_var.set("No frame. Check cameras.")
            self.root.after(30, self.update_frame)
            return

        if not ok_line and ok_ball:
            frame_line = frame_ball
            ok_line = True
        if not ok_ball and ok_line:
            frame_ball = frame_line
            ok_ball = True

        # Apply adjustable parameters
        self.line_detector.black_v_max = int(self.line_v_max.get())
        self.line_detector.black_s_max = int(self.line_s_max.get())
        self.line_detector.min_black_area = int(self.line_min_area.get())
        self.line_detector.erode_iter = int(self.line_erode.get())
        self.line_detector.dilate_iter = int(self.line_dilate.get())

        self.ball_detector.SILVER_BLUR = int(self.silver_blur.get())
        self.ball_detector.SILVER_HOUGH_DP = float(self.silver_dp.get())
        self.ball_detector.SILVER_HOUGH_MIN_DISTANCE = int(self.silver_min_dist.get())
        self.ball_detector.SILVER_HOUGH_PARAM1 = int(self.silver_p1.get())
        self.ball_detector.SILVER_HOUGH_PARAM2 = int(self.silver_p2.get())
        self.ball_detector.SILVER_HOUGH_MIN_RADIUS = int(self.silver_min_r.get())
        self.ball_detector.SILVER_HOUGH_MAX_RADIUS = int(self.silver_max_r.get())
        b = int(self.dead_black_thresh.get())
        self.ball_detector.DEAD_BLACK_THRESHOLD = (b, b, b)
        self.ball_detector.HOUGH_PARAMETER_1 = int(self.hough_p1.get())
        self.ball_detector.HOUGH_PARAMETER_2 = int(self.hough_p2.get())
        self.ball_detector.HOUGH_MIN_RADIUS = int(self.hough_min_r.get())
        self.ball_detector.HOUGH_MAX_RADIUS = int(self.hough_max_r.get())

        # Line detection
        line_w = int(self.line_width.get())
        line_h = int(self.line_height.get())
        self.line_detector.WIDTH = line_w
        self.line_detector.HEIGHT = line_h
        line_frame = cv2.resize(frame_line, (line_w, line_h))
        if self.line_use_transform.get():
            line_view = perspective_transform(line_frame, line_w, line_h)
        else:
            line_view = line_frame
        line_display = line_view.copy()

        contour = None
        try:
            contour, _ = self.line_detector.black_mask(line_view, line_display)
            if contour is not None:
                angle, gap = self.line_detector.calculate_angle(contour, line_display)
                cv2.putText(
                    line_display,
                    f"ANGLE: {angle}",
                    (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                if gap > 0:
                    cv2.putText(
                        line_display,
                        f"GAP: {gap}",
                        (5, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
            else:
                cv2.putText(
                    line_display,
                    "LINE NOT FOUND",
                    (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
        except Exception as exc:
            cv2.putText(
                line_display,
                f"LINE ERR: {exc}",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        silver_found = False
        if self.silver_detector is not None and self.silver_enabled.get():
            try:
                result = self.silver_detector.predict(line_view)
                if isinstance(result, tuple):
                    result = result[0]
                if result["prediction"] == 1 and result["confidence"] >= float(self.silver_conf.get()):
                    silver_found = True
                    cv2.putText(
                        line_display,
                        f"SILVER LINE {result['confidence']:.2f}",
                        (5, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
            except Exception as exc:
                cv2.putText(
                    line_display,
                    f"SILVER ERR: {exc}",
                    (5, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Balls detection
        ball_frame = cv2.resize(frame_ball, (320, 240))
        ball_display = ball_frame.copy()

        live_x = self.ball_detector.live(ball_frame, ball_display, self.last_live_x, draw=False)
        dead_x = self.ball_detector.dead(ball_frame, ball_display, self.last_dead_x)

        if live_x is not None and dead_x is not None:
            overlap_px = int(self.silver_black_overlap.get())
            if abs(live_x - dead_x) <= overlap_px:
                live_x = None

        if live_x is not None:
            circle = self.ball_detector.last_live_circle
            if circle is not None:
                cx, cy, cr = circle
                cv2.circle(ball_display, (cx, cy), cr, (0, 255, 255), 2)
                cv2.circle(ball_display, (cx, cy), 2, (0, 255, 255), 2)
            cv2.putText(
                ball_display,
                "SILVER BALL",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.line(ball_display, (live_x, 0), (live_x, ball_display.shape[0] - 1), (0, 255, 255), 1)
            self.last_live_x = live_x

        if dead_x is not None:
            cv2.putText(
                ball_display,
                "BLACK BALL",
                (5, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.line(ball_display, (dead_x, 0), (dead_x, ball_display.shape[0] - 1), (0, 0, 0), 2)
            self.last_dead_x = dead_x

        # Status strip (top-left)
        status_parts = []
        status_parts.append("LINE OK" if contour is not None else "LINE NO")
        status_parts.append("SILVER LINE" if silver_found else "NO SILVER LINE")
        status_parts.append("SILVER BALL" if live_x is not None else "NO SILVER BALL")
        status_parts.append("BLACK BALL" if dead_x is not None else "NO BLACK BALL")
        status_text = " | ".join(status_parts)
        cv2.putText(
            line_display,
            status_text,
            (5, line_display.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

        # Composite view (fit to camera size)
        base_w = int(self.camera_width.get())
        base_h = int(self.camera_height.get())
        try:
            avail_w = self.root.winfo_width() - self.ctrl_canvas.winfo_width() - self.ctrl_scroll.winfo_width() - 32
            avail_h = self.root.winfo_height() - 32
            if avail_w < 200 or avail_h < 200:
                avail_w, avail_h = base_w, base_h
        except Exception:
            avail_w, avail_h = base_w, base_h

        display_w = max(base_w, avail_w)
        display_h = max(base_h, avail_h)

        half_w = max(160, display_w // 2)
        line_disp = cv2.resize(line_display, (half_w, display_h))
        ball_disp = cv2.resize(ball_display, (display_w - half_w, display_h))
        combined = np.hstack([line_disp, ball_disp])
        combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(combined)
        self.photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.photo)

        # FPS
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            fps = self.frame_count / (now - self.last_fps_time)
            self.fps_var.set(f"FPS: {fps:.1f}")
            self.frame_count = 0
            self.last_fps_time = now

        self.root.after(10, self.update_frame)

    def on_close(self):
        self.running = False
        try:
            for cap in (self.cap_line, self.cap_ball):
                if cap is not None:
                    cap.release()
        except Exception:
            pass
        self.root.destroy()


def main():
    app = VisionUI()
    app.root.mainloop()


if __name__ == "__main__":
    main()
