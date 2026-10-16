#!/usr/bin/env python3
"""
Drone Perspective (FPV) Renderer with Empirical YOLO Noise & Kalman Tracking.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Generates a broadcast-quality on-board camera video (640x480 FPV sensor) showing:
1. Ground-Truth 3D Target Projection (thin dashed box)
2. Raw Empirical YOLOv8 Measurement (yellow box with noise, or DROPOUT indicator)
3. Kalman Filter State Estimate (Neon Green locked / Amber dead-reckoning coasting)
4. Full Telemetry HUD, Artificial Horizon, and 2D Tactical Sky Radar Inset.
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.simulation import FastPixhawkQuadSim
from control.pid_controller import KinematicVisualServoController
from control.trajectory import StochasticTargetTrajectory
from control.synthetic_sensor import SyntheticYOLOSensor
from perception.pipeline import DroneTracker


def render_empirical_fpv(
    output_mp4: Path,
    output_gif: Optional[Path] = None,
    duration: float = 12.0,
    profile: str = "evasive",
    env_mode: str = "nominal",
    seed: int = 42,
    img_w: int = 640,
    img_h: int = 480,
    fps: int = 30,
):
    print(f"\n{'='*75}")
    print(f"  RENDERING DRONE CAMERA VIEW (FPV): EMPIRICAL YOLO + KALMAN TRACKING")
    print(f"  Duration: {duration:.1f}s | FPS: {fps} | Profile: '{profile}' | Env Mode: '{env_mode}'")
    print(f"  Output MP4: {output_mp4}")
    if output_gif:
        print(f"  Output GIF: {output_gif}")
    print(f"{'='*75}\n")

    dt_sim = 0.02  # 50 Hz guidance control loop
    camera_dt = 1.0 / 30.0  # 30 Hz camera frame update
    total_sim_steps = int(round(duration / dt_sim))

    # Initialize Simulator, Target, Sensor & Kalman Tracker
    sim = FastPixhawkQuadSim(dt=dt_sim)
    sim.reset(initial_pos=[0.0, 0.0, -2.5], initial_yaw=0.0)

    target_oracle = StochasticTargetTrajectory(
        duration=duration + 10.0,
        profile=profile,
        initial_pos=np.array([6.0, 0.5, -2.5]),  # Start at ~6m standoff in front
        seed=seed
    )

    sensor = SyntheticYOLOSensor(
        img_w=img_w,
        img_h=img_h,
        env_mode=env_mode,
        seed=seed
    )

    tracker = DroneTracker(
        dt=dt_sim,
        max_lost_frames=int(round(3.5 / dt_sim)),
        min_hits_to_confirm=2
    )

    controller = KinematicVisualServoController(
        camera_uptilt_deg=15.0,
        hfov_deg=60.0,
        vfov_deg=45.0,
        desired_bbox_size=32.33,
        desired_standoff_dist=6.0,
        use_bbox_size=False,
        enable_lateral_strafe=True,
        kp_lat=2.5,
        kd_lat=0.35,
        max_lat_vel=4.0,
        max_lat_accel=5.0,
        max_limits=np.array([18.0, 4.0, 2.5, 120.0]),
        max_accel=6.5,
        max_decel=2.8,
    )

    # Video Canvas dimensions: 1280 x 720 (720p HD broadcast)
    # Left / Main Canvas: 854 x 720 (Expanded FPV camera view)
    # Right Sidebar: 426 x 720 (Tactical Radar + Telemetry Metrics)
    canvas_w = 1280
    canvas_h = 720
    fpv_view_w = 854
    fpv_view_h = 720

    # FFmpeg pipe setup for pristine MP4 output
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{canvas_w}x{canvas_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "18",
        str(output_mp4)
    ]
    pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    time_since_camera = 0.0
    latest_det_box = None
    latest_det_conf = 0.0
    latest_yolo_status = "INITIALIZING"
    yolo_dropout_frames = 0
    yolo_total_detections = 0
    yolo_total_frames = 0
    kalman_coasting_steps = 0

    recorded_frames = []
    render_interval = 1.0 / fps
    time_since_render = 0.0

    print("[SIM] Running closed-loop simulation and rendering frames...")

    for step in range(total_sim_steps):
        t = step * dt_sim
        time_since_camera += dt_sim
        time_since_render += dt_sim

        p_target = target_oracle.get_position(t)

        # 1. Perspective Projection of 3D target onto camera sensor
        raw_ex, raw_ey, raw_in_view, dist, raw_bw, raw_bh = sim.project_to_camera(
            p_target, return_bbox=True, img_w=img_w, img_h=img_h
        )

        gt_box = None
        if raw_in_view and raw_bw > 1.0 and raw_bh > 1.0:
            gt_cx = (raw_ex + 1.0) * img_w / 2.0
            gt_cy = (raw_ey + 1.0) * img_h / 2.0
            gt_box = np.array([
                gt_cx - raw_bw / 2.0,
                gt_cy - raw_bh / 2.0,
                gt_cx + raw_bw / 2.0,
                gt_cy + raw_bh / 2.0
            ], dtype=np.float32)

        # 2. Camera Frame Update (30 Hz)
        is_camera_tick = (time_since_camera >= camera_dt - 1e-5)
        if is_camera_tick:
            time_since_camera -= camera_dt
            yolo_total_frames += 1

            norm_cx = (raw_ex + 1.0) / 2.0
            norm_cy = (raw_ey + 1.0) / 2.0
            norm_w  = raw_bw / img_w
            norm_h  = raw_bh / img_h

            det_bbox, conf, fps_boxes = sensor.step(
                norm_cx, norm_cy, norm_w, norm_h, in_fov=raw_in_view
            )

            detections = []
            confs = []
            if det_bbox is not None:
                detections.append(det_bbox)
                confs.append(conf)
                latest_det_box = det_bbox.copy()
                latest_det_conf = conf
                latest_yolo_status = "DETECTED"
                yolo_dropout_frames = 0
                yolo_total_detections += 1
            else:
                latest_det_box = None
                latest_det_conf = 0.0
                latest_yolo_status = "DROPOUT / MISS"
                yolo_dropout_frames += 1

            for fp in fps_boxes:
                detections.append(fp)
                confs.append(0.36)

            det_arr = np.array(detections, dtype=np.float32) if detections else np.empty((0, 4), dtype=np.float32)
            conf_arr = np.array(confs, dtype=np.float32) if confs else np.empty((0,), dtype=np.float32)
            tracker.update(det_arr, conf_arr)
        else:
            # 50 Hz dead-reckoning prediction between camera updates
            for trk in tracker.trackers:
                trk.predict()

        # 3. Retrieve primary Kalman track
        primary = None
        if tracker.primary_track_id is not None:
            primary = next((t for t in tracker.trackers if t.id == tracker.primary_track_id), None)

        kalman_box = None
        kalman_vel = (0.0, 0.0)
        kalman_is_coasting = True
        kalman_hits = 0

        if primary is not None and primary.hits >= tracker.min_hits_to_confirm:
            kalman_box = primary.get_state()
            kalman_vel = primary.get_velocity()
            kalman_is_coasting = (primary.time_since_update > 0)
            kalman_hits = primary.hits
            if kalman_is_coasting:
                kalman_coasting_steps += 1
            else:
                kalman_coasting_steps = 0

            cx = (kalman_box[0] + kalman_box[2]) / 2.0
            cy = (kalman_box[1] + kalman_box[3]) / 2.0
            bw_est = max(1.0, kalman_box[2] - kalman_box[0])
            bh_est = max(1.0, kalman_box[3] - kalman_box[1])
            ex_est = (cx - img_w / 2.0) / (img_w / 2.0)
            ey_est = (cy - img_h / 2.0) / (img_h / 2.0)

            telemetry = {
                "error_x": ex_est,
                "error_y": ey_est,
                "target_size": bw_est,
                "distance": dist,
                "in_view": True,
                "status": "TRACKING" if not kalman_is_coasting else "COASTING"
            }
        else:
            telemetry = {
                "error_x": 0.0,
                "error_y": 0.0,
                "target_size": 32.33,
                "distance": 6.0,
                "in_view": False,
                "status": "SEARCHING"
            }

        # 4. Guidance controller step (50 Hz)
        cmd = controller.compute_cmd(telemetry, drone_pitch=sim.pitch, drone_roll=sim.roll, dt=dt_sim)
        sim.step(cmd)

        # 5. Render Video Frame (at fps rate)
        if time_since_render >= render_interval - 1e-5:
            time_since_render -= render_interval

            frame = draw_composite_frame(
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                fpv_w=fpv_view_w,
                fpv_h=fpv_view_h,
                sensor_w=img_w,
                sensor_h=img_h,
                t=t,
                sim=sim,
                p_target=p_target,
                gt_box=gt_box,
                yolo_box=latest_det_box,
                yolo_conf=latest_det_conf,
                yolo_status=latest_yolo_status,
                yolo_drop_burst=yolo_dropout_frames,
                kalman_box=kalman_box,
                kalman_vel=kalman_vel,
                kalman_coasting=kalman_is_coasting,
                kalman_coasting_steps=kalman_coasting_steps,
                kalman_hits=kalman_hits,
                dist=dist,
                cmd=cmd,
                yolo_det_rate=(yolo_total_detections / max(1, yolo_total_frames)) * 100.0,
                primary_track=primary
            )

            pipe.stdin.write(frame.tobytes())
            if output_gif and len(recorded_frames) < 180 and (len(recorded_frames) % 2 == 0):
                # Save lower-res preview frame for animated GIF (15 fps, 640x360)
                small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                recorded_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

    pipe.stdin.close()
    pipe.wait()
    print(f"✓ Successfully rendered video to: {output_mp4}")

    # Generate animated GIF if requested
    if output_gif and output_mp4.exists():
        print(f"[GIF] Compiling animated GIF using ffmpeg...")
        gif_cmd = [
            "ffmpeg", "-y",
            "-i", str(output_mp4),
            "-vf", "fps=15,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            str(output_gif)
        ]
        subprocess.run(gif_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✓ Successfully saved animated GIF: {output_gif}")


def draw_composite_frame(
    canvas_w: int,
    canvas_h: int,
    fpv_w: int,
    fpv_h: int,
    sensor_w: int,
    sensor_h: int,
    t: float,
    sim: FastPixhawkQuadSim,
    p_target: np.ndarray,
    gt_box: Optional[np.ndarray],
    yolo_box: Optional[np.ndarray],
    yolo_conf: float,
    yolo_status: str,
    yolo_drop_burst: int,
    kalman_box: Optional[np.ndarray],
    kalman_vel: Tuple[float, float],
    kalman_coasting: bool,
    kalman_coasting_steps: int,
    kalman_hits: int,
    dist: float,
    cmd: np.ndarray,
    yolo_det_rate: float,
    primary_track: Optional[object]
) -> np.ndarray:
    """Draws the complete 1280x720 broadcast frame with FPV Viewfinder and Tactical HUD."""
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (14, 17, 23)  # Dark slate background

    # -------------------------------------------------------------
    # 1. LEFT PANE: DRONE CAMERA FPV VIEWFINDER (640x480 scaled to fpv_w x fpv_h)
    # -------------------------------------------------------------
    scale_x = fpv_w / float(sensor_w)
    scale_y = fpv_h / float(sensor_h)

    def to_fpv(x_px: float, y_px: float) -> Tuple[int, int]:
        return int(round(x_px * scale_x)), int(round(y_px * scale_y))

    # FPV Background: Sky/Ground gradient tilted by roll & pitch
    fpv_img = np.zeros((fpv_h, fpv_w, 3), dtype=np.uint8)
    fpv_img[:] = (20, 24, 32)  # High-tech night cockpit tint

    # Horizon line
    roll = sim.roll
    cam_pitch = sim.pitch + sim.camera_uptilt
    horiz_y_center = (sensor_h / 2.0) + (np.tan(cam_pitch) / np.tan(np.deg2rad(22.5))) * (sensor_h / 2.0)
    tan_roll = np.tan(-roll)
    y_l = int(round((horiz_y_center - (sensor_w / 2.0) * tan_roll) * scale_y))
    y_r = int(round((horiz_y_center + (sensor_w / 2.0) * tan_roll) * scale_y))
    cv2.line(fpv_img, (0, y_l), (fpv_w, y_r), (70, 90, 120), 1, cv2.LINE_AA)

    # Pitch Ladder Ticks (+10° and -10°)
    for deg in (-10, 10):
        deg_rad = np.deg2rad(deg)
        ladder_y = (sensor_h / 2.0) + (np.tan(cam_pitch + deg_rad) / np.tan(np.deg2rad(22.5))) * (sensor_h / 2.0)
        yl_l = int(round((ladder_y - 30.0 * tan_roll) * scale_y))
        yl_r = int(round((ladder_y + 30.0 * tan_roll) * scale_y))
        cv2.line(fpv_img, (int(fpv_w / 2 - 40), yl_l), (int(fpv_w / 2 + 40), yl_r), (60, 75, 95), 1, cv2.LINE_AA)
        cv2.putText(fpv_img, f"{deg:+d}", (int(fpv_w / 2 + 46), yl_r + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 100, 125), 1, cv2.LINE_AA)

    # Optical Boresight Crosshairs
    cx_s, cy_s = int(fpv_w / 2), int(fpv_h / 2)
    reticle_col = (0, 210, 210)
    cv2.circle(fpv_img, (cx_s, cy_s), 16, reticle_col, 1, cv2.LINE_AA)
    cv2.line(fpv_img, (cx_s - 30, cy_s), (cx_s - 20, cy_s), reticle_col, 1, cv2.LINE_AA)
    cv2.line(fpv_img, (cx_s + 20, cy_s), (cx_s + 30, cy_s), reticle_col, 1, cv2.LINE_AA)
    cv2.line(fpv_img, (cx_s, cy_s - 30), (cx_s, cy_s - 20), reticle_col, 1, cv2.LINE_AA)
    cv2.line(fpv_img, (cx_s, cy_s + 20), (cx_s, cy_s + 30), reticle_col, 1, cv2.LINE_AA)

    # 6.0m Goal Standoff Target Reference Ring (32.3px width)
    ref_rad = int(round((32.33 / 2.0) * scale_x))
    cv2.circle(fpv_img, (cx_s, cy_s), ref_rad, (0, 140, 140), 1, cv2.LINE_AA)

    # A. Draw Ground-Truth 3D Bounding Box (Subtle Thin Dashed White/Green)
    if gt_box is not None:
        gx1, gy1 = to_fpv(gt_box[0], gt_box[1])
        gx2, gy2 = to_fpv(gt_box[2], gt_box[3])
        cv2.rectangle(fpv_img, (gx1, gy1), (gx2, gy2), (120, 150, 120), 1, cv2.LINE_AA)
        cv2.putText(fpv_img, "GT", (gx1, gy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 150, 120), 1, cv2.LINE_AA)

    # B. Draw Raw YOLO Measurement Box (Vibrant Yellow when detected, Red Dropout Alert when missed)
    if yolo_box is not None:
        yx1, yy1 = to_fpv(yolo_box[0], yolo_box[1])
        yx2, yy2 = to_fpv(yolo_box[2], yolo_box[3])
        # Yellow bounding box for raw YOLO observation
        y_col = (0, 230, 255)
        cv2.rectangle(fpv_img, (yx1, yy1), (yx2, yy2), y_col, 2, cv2.LINE_AA)
        cv2.putText(
            fpv_img, f"YOLOv8 [{yolo_conf*100:.0f}%]",
            (yx1, yy2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, y_col, 1, cv2.LINE_AA
        )
    else:
        # Visual notification of YOLO Dropout Burst
        cv2.putText(
            fpv_img, f"● YOLO DROPOUT BURST ({yolo_drop_burst} frames / {yolo_drop_burst*33.3:.0f}ms)",
            (24, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (60, 60, 255), 1, cv2.LINE_AA
        )

    # C. Draw Kalman Filter Estimated Track (Neon Green when Locked / Amber when Coasting)
    if kalman_box is not None:
        kx1, ky1 = to_fpv(kalman_box[0], kalman_box[1])
        kx2, ky2 = to_fpv(kalman_box[2], kalman_box[3])
        kcx, kcy = int((kx1 + kx2) / 2), int((ky1 + ky2) / 2)

        if kalman_coasting:
            k_col = (0, 165, 255)  # Amber: Coasting / Dead Reckoning
            status_text = f"TRACK #1 [COASTING +{kalman_coasting_steps*20}ms]"
        else:
            k_col = (0, 255, 120)  # Neon Green: Actively Confirmed
            status_text = f"LOCKED TARGET #1 (Hits: {kalman_hits})"

        # Corner Brackets
        arm_len = max(8, min(24, int((kx2 - kx1) * 0.28)))
        t_thick = 3 if not kalman_coasting else 2
        # TL
        cv2.line(fpv_img, (kx1, ky1), (kx1 + arm_len, ky1), k_col, t_thick, cv2.LINE_AA)
        cv2.line(fpv_img, (kx1, ky1), (kx1, ky1 + arm_len), k_col, t_thick, cv2.LINE_AA)
        # TR
        cv2.line(fpv_img, (kx2, ky1), (kx2 - arm_len, ky1), k_col, t_thick, cv2.LINE_AA)
        cv2.line(fpv_img, (kx2, ky1), (kx2, ky1 + arm_len), k_col, t_thick, cv2.LINE_AA)
        # BL
        cv2.line(fpv_img, (kx1, ky2), (kx1 + arm_len, ky2), k_col, t_thick, cv2.LINE_AA)
        cv2.line(fpv_img, (kx1, ky2), (kx1, ky2 - arm_len), k_col, t_thick, cv2.LINE_AA)
        # BR
        cv2.line(fpv_img, (kx2, ky2), (kx2 - arm_len, ky2), k_col, t_thick, cv2.LINE_AA)
        cv2.line(fpv_img, (kx2, ky2), (kx2, ky2 - arm_len), k_col, t_thick, cv2.LINE_AA)

        # Kalman Filter Velocity Vector Arrow
        vx_px, vy_px = kalman_vel
        arrow_len = 0.4
        arrow_end = (int(kcx + vx_px * scale_x * arrow_len), int(kcy + vy_px * scale_y * arrow_len))
        cv2.arrowedLine(fpv_img, (kcx, kcy), arrow_end, (255, 255, 0), 2, cv2.LINE_AA, tipLength=0.3)

        # Track Label Badge
        cv2.rectangle(fpv_img, (kx1, ky1 - 20), (kx1 + 220, ky1), (10, 14, 18), -1)
        cv2.rectangle(fpv_img, (kx1, ky1 - 20), (kx1 + 220, ky1), k_col, 1)
        cv2.putText(fpv_img, status_text, (kx1 + 6, ky1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, k_col, 1, cv2.LINE_AA)

        # Breadcrumb trail from Kalman tracker history
        if primary_track is not None and hasattr(primary_track, "history"):
            pts = [to_fpv(pt[0], pt[1]) for pt in primary_track.history]
            for i in range(1, len(pts)):
                cv2.line(fpv_img, pts[i - 1], pts[i], k_col, 1, cv2.LINE_AA)

    # Top FPV Watermark HUD
    cv2.putText(
        fpv_img, "ON-BOARD CAMERA VIEW (30 Hz SENSOR / 50 Hz KALMAN DEAD RECKONING)",
        (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 210, 210), 1, cv2.LINE_AA
    )
    cv2.putText(
        fpv_img, f"OPTICAL ENVELOPE: 60° HFOV x 45° VFOV | TILT: +15° | TIME: {t:04.1f}s",
        (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 180, 200), 1, cv2.LINE_AA
    )

    # Legend at bottom of FPV view
    cv2.rectangle(fpv_img, (0, fpv_h - 40), (fpv_w, fpv_h), (8, 10, 14), -1)
    cv2.line(fpv_img, (0, fpv_h - 40), (fpv_w, fpv_h - 40), (40, 50, 60), 1)
    leg_y = fpv_h - 16
    # GT
    cv2.rectangle(fpv_img, (20, leg_y - 10), (34, leg_y + 2), (120, 150, 120), 1)
    cv2.putText(fpv_img, "Ground-Truth 3D", (40, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 160, 140), 1, cv2.LINE_AA)
    # YOLO
    cv2.rectangle(fpv_img, (180, leg_y - 10), (194, leg_y + 2), (0, 230, 255), 2)
    cv2.putText(fpv_img, "Empirical YOLOv8", (200, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 255), 1, cv2.LINE_AA)
    # Kalman Locked
    cv2.rectangle(fpv_img, (360, leg_y - 10), (374, leg_y + 2), (0, 255, 120), 2)
    cv2.putText(fpv_img, "Kalman Locked", (380, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 120), 1, cv2.LINE_AA)
    # Kalman Coasting
    cv2.rectangle(fpv_img, (520, leg_y - 10), (534, leg_y + 2), (0, 165, 255), 2)
    cv2.putText(fpv_img, "Kalman Coasting (Dead Reckoning)", (540, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 165, 255), 1, cv2.LINE_AA)

    canvas[0:fpv_h, 0:fpv_w] = fpv_img

    # Vertical dividing line
    cv2.line(canvas, (fpv_w, 0), (fpv_w, canvas_h), (50, 60, 75), 2)

    # -------------------------------------------------------------
    # 2. RIGHT PANE: TACTICAL SKY RADAR (TOP) & TELEMETRY HUD (BOTTOM)
    # -------------------------------------------------------------
    sidebar_x = fpv_w + 1
    sidebar_w = canvas_w - sidebar_x

    # Tactical Radar Inset (Top-Right: 426 x 320)
    radar_h = 310
    radar_img = np.zeros((radar_h, sidebar_w, 3), dtype=np.uint8)
    radar_img[:] = (10, 13, 18)

    # Radar Title
    cv2.rectangle(radar_img, (0, 0), (sidebar_w, 28), (20, 26, 36), -1)
    cv2.putText(radar_img, "TACTICAL RADAR (TOP-DOWN 2D)", (14, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 210, 210), 1, cv2.LINE_AA)

    # Radar Coordinate Mapping (Chaser centered, oriented along heading)
    rcx = int(sidebar_w / 2)
    rcy = int(radar_h / 2 + 15)
    r_scale = 16.0  # pixels per meter (covers ~20m range)

    # Range rings (5m, 10m, 15m)
    for r_m in (5, 6, 7, 10):
        r_px = int(r_m * r_scale)
        col = (0, 180, 0) if (5 <= r_m <= 7) else (40, 50, 60)
        thick = 2 if r_m == 6 else 1
        cv2.circle(radar_img, (rcx, rcy), r_px, col, thick, cv2.LINE_AA)
        cv2.putText(radar_img, f"{r_m}m", (rcx + 4, rcy - r_px + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (70, 85, 100), 1, cv2.LINE_AA)

    # FOV Cone (60° HFOV radiating from chaser)
    half_fov = np.deg2rad(30.0)
    cone_len = 14.0 * r_scale
    cone_l = (int(rcx - cone_len * np.sin(half_fov)), int(rcy - cone_len * np.cos(half_fov)))
    cone_r = (int(rcx + cone_len * np.sin(half_fov)), int(rcy - cone_len * np.cos(half_fov)))
    cv2.line(radar_img, (rcx, rcy), cone_l, (0, 160, 200), 1, cv2.LINE_AA)
    cv2.line(radar_img, (rcx, rcy), cone_r, (0, 160, 200), 1, cv2.LINE_AA)

    # Relative target position in chaser body frame
    rel_w = p_target - sim.pos
    cy_yaw, sy_yaw = np.cos(sim.yaw), np.sin(sim.yaw)
    rel_fwd = cy_yaw * rel_w[0] + sy_yaw * rel_w[1]
    rel_right = -sy_yaw * rel_w[0] + cy_yaw * rel_w[1]

    t_px = int(rcx + rel_right * r_scale)
    t_py = int(rcy - rel_fwd * r_scale)

    # Target glyph on radar
    cv2.circle(radar_img, (t_px, t_py), 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(radar_img, (t_px, t_py), 9, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(radar_img, f"TGT ({dist:.1f}m)", (t_px + 10, t_py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)

    # Chaser glyph on radar
    cv2.circle(radar_img, (rcx, rcy), 5, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.line(radar_img, (rcx, rcy), (rcx, rcy - 16), (0, 255, 0), 2, cv2.LINE_AA)

    canvas[0:radar_h, sidebar_x:canvas_w] = radar_img
    cv2.line(canvas, (sidebar_x, radar_h), (canvas_w, radar_h), (50, 60, 75), 1)

    # Telemetry HUD Card (Bottom-Right: 426 x 410)
    hud_y_start = radar_h + 1
    hud_h = canvas_h - hud_y_start
    hud_img = np.zeros((hud_h, sidebar_w, 3), dtype=np.uint8)
    hud_img[:] = (12, 15, 22)

    cv2.rectangle(hud_img, (0, 0), (sidebar_w, 28), (20, 26, 36), -1)
    cv2.putText(hud_img, "AUTOPILOT & PERCEPTION METRICS", (14, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 210, 210), 1, cv2.LINE_AA)

    lines = [
        ("TIME ELAPSED", f"{t:04.1f} s / 12.0 s", (220, 220, 220)),
        ("STANDOFF DISTANCE", f"{dist:5.2f} m (Goal: 6.00m)", (0, 255, 120) if (5.0 <= dist <= 7.0) else (0, 165, 255)),
        ("IN-BASKET LOCK", "ENGAGED (5-7m FOV)" if (5.0 <= dist <= 7.0) else "ACQUIRING", (0, 255, 120) if (5.0 <= dist <= 7.0) else (80, 80, 240)),
        ("--------------------", "------------------------", (50, 60, 75)),
        ("CAMERA SENSOR", "30 Hz Global Shutter", (180, 190, 200)),
        ("YOLO DETECTION", yolo_status, (0, 230, 255) if yolo_status == "DETECTED" else (60, 60, 255)),
        ("DETECTION RATE", f"{yolo_det_rate:.1f}% (Empirical Match)", (180, 190, 200)),
        ("KALMAN TRACKER", "CONFIRMED LOCK" if not kalman_coasting else "DEAD RECKONING", (0, 255, 120) if not kalman_coasting else (0, 165, 255)),
        ("COASTING DURATION", f"+{kalman_coasting_steps*20:3d} ms" if kalman_coasting else "0 ms (MEASUREMENT FRESH)", (0, 165, 255) if kalman_coasting else (120, 140, 160)),
        ("--------------------", "------------------------", (50, 60, 75)),
        ("GUIDANCE LOOP", "50 Hz Closed-Loop", (180, 190, 200)),
        ("FWD SURGE SPEED", f"{cmd[0]:4.1f} m/s", (220, 220, 220)),
        ("LATERAL STRAFE", f"{cmd[1]:+4.2f} m/s", (220, 220, 220)),
        ("VERTICAL HEAVE", f"{-cmd[2]:+4.2f} m/s", (220, 220, 220)),
        ("YAW TURN RATE", f"{cmd[3]:+5.1f} °/s", (220, 220, 220)),
        ("CHASER ROLL / PITCH", f"{np.rad2deg(sim.roll):+4.1f}° / {np.rad2deg(sim.pitch):+4.1f}°", (160, 180, 200)),
    ]

    curr_y = 50
    for label, val, val_col in lines:
        if label.startswith("---"):
            cv2.line(hud_img, (14, curr_y - 4), (sidebar_w - 14, curr_y - 4), val_col, 1)
            curr_y += 12
            continue
        cv2.putText(hud_img, label, (14, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (130, 145, 165), 1, cv2.LINE_AA)
        cv2.putText(hud_img, val, (180, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, val_col, 1, cv2.LINE_AA)
        curr_y += 22

    canvas[hud_y_start:canvas_h, sidebar_x:canvas_w] = hud_img

    return canvas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Drone Perspective (FPV) Video with Empirical YOLO Noise & Kalman Filter.")
    parser.add_argument("--duration", type=float, default=12.0, help="Simulation duration (seconds)")
    parser.add_argument("--profile", type=str, default="evasive", help="Target flight profile (cruising, evasive, aerobatic)")
    parser.add_argument("--env-mode", type=str, default="nominal", help="Perception clutter mode (nominal, cluttered, none)")
    parser.add_argument("--output-mp4", type=str, default="drone_fpv_empirical_chase.mp4", help="Output MP4 filename")
    parser.add_argument("--output-gif", type=str, default="drone_fpv_empirical_chase.gif", help="Output animated GIF filename")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_mp4 = Path(args.output_mp4).resolve()
    out_gif = Path(args.output_gif).resolve() if args.output_gif else None

    render_empirical_fpv(
        output_mp4=out_mp4,
        output_gif=out_gif,
        duration=args.duration,
        profile=args.profile,
        env_mode=args.env_mode,
        seed=args.seed
    )
