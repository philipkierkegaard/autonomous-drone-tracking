# Drone Deployment Pre-Flight Readiness Checklist
**Project:** Autonomous Drone Pursuit & Vision-Based Interception  
**Target Platform:** NVIDIA Jetson Orin Nano + Pixhawk (ArduCopter 4.x via PyMAVLink)  
**Status:** Living Document (Updated Iteratively Before Flight Trials)

---

## 📋 Executive Overview Table

| ID | Priority | Status | Component | Problem Description (Single Sentence) |
| :---: | :---: | :---: | :--- | :--- |
| **1.1** | 🔴 P0 | `[ ] Pending` | [`autopilot_bridge.py`](control/autopilot_bridge.py) | Non-finite (`NaN`/`Inf`) image errors propagate directly into MAVLink velocity packets without sanity checks. |
| **1.2** | 🔴 P0 | `[ ] Pending` | [`run_tracker.py`](run_tracker.py) | The system fails to distinguish between unknown altitude and low altitude, commanding an emergency climb on the ground. |
| **1.3** | 🔴 P0 | `[ ] Pending` | [`autopilot_bridge.py`](control/autopilot_bridge.py) | Telemetry freshness is not verified, causing the controller to fly on frozen attitude/altitude if MAVLink streams stall. |
| **1.4** | 🔴 P0 | `[ ] Pending` | [`run_tracker.py`](run_tracker.py) | Downward distance sensor lacks timestamp expiration and orientation filtering, risking stale ground locks. |
| **1.5** | 🔴 P0 | `[ ] Pending` | [`run_tracker.py`](run_tracker.py) | Video frames are timestamped after GPU inference finishes rather than at physical camera capture, masking aged frames. |
| **1.6** | 🔴 P0 | `[ ] Pending` | [`run_tracker.py`](run_tracker.py) | In legacy coast mode, forward velocity does not decay progressively, while active brake mode relies on outer slew limits. |
| **2.1** | 🟡 P1 | `[ ] Pending` | [`pipeline.py`](perception/pipeline.py) | The Kalman tracker assumes a hardcoded 30 FPS interval ($dt=1/30\text{s}$) rather than using dynamically measured frame intervals. |
| **2.2** | 🟡 P1 | `[ ] Pending` | [`pipeline.py`](perception/pipeline.py) | A new primary target lock can acquire an already coasting dead-reckoned track without an active visual detection. |
| **2.3** | 🟡 P1 | `[ ] Pending` | [`run_tracker.py`](run_tracker.py) | Candidate model weights list defaults to the older CIoU baseline, omitting the latest calibrated Hybrid NWD models. |
| **2.4** | 🟡 P1 | `[ ] Pending` | [`spatial.py`](control/spatial.py) | Camera FOV and mount uptilt CLI overrides reach the controller but are ignored by ego-motion compensation defaults. |
| **3.1** | 🟢 P2 | `[ ] Pending` | [`run_tracker.py`](run_tracker.py) | Synchronous video encoding and GUI rendering execute on the same asyncio thread as the 50 Hz MAVLink flight loop. |
| **3.2** | 🟢 P2 | `[ ] Pending` | [`run_tracker.py`](run_tracker.py) | Hardcoded 35 ms camera latency lookup in vision worker is redundant if translational/rotational ego-compensation is disabled. |
| **3.3** | ⚪ Backlog | `[-] Deprioritized`| [`pipeline.py`](perception/pipeline.py) | Target identity can switch between adjacent visual objects without appearance re-identification (deferred for single-target tests). |

---

## 🔴 Tier 1: Critical Flight-Safety Blockers (Must Fix Before Props-On)

### [ ] Item 1.1: Sanitize Non-Finite (`NaN` / `Inf`) Telemetry Before MAVLink Dispatch
* **Locations:**
  * [`control/autopilot_bridge.py:L743`](control/autopilot_bridge.py#L743) (`send_velocity_body`)
  * [`control/pid_controller.py:L310`](control/pid_controller.py#L310) (`compute_cmd`)
  * [`control/spatial.py:L51`](control/spatial.py#L51) (`unproject_camera_to_horizon`)
* **Description:**
  If an image bounding box calculation, lost detection, or math error produces a `NaN`, `np.clip` fails silently. `send_velocity_body` packs `nan` directly into the MAVLink `SET_POSITION_TARGET_LOCAL_NED` packet. In ArduPilot GUIDED mode, sending `NaN` velocity setpoints can corrupt internal EKF velocity estimators or trigger uncommanded motor behavior.
* **Possible Fix:**
  Add a strict sanity gate at the entry of `send_velocity_body`:
  ```python
  if not all(math.isfinite(v) for v in (v_fwd, v_right, v_down, yawspeed_deg_s)):
      # Drop command and command a safe zero-velocity holding hover
      return self.send_hold_hover()
  ```
  And in `control/pid_controller.py`:
  ```python
  if not all(np.isfinite([err_x, err_y, range_term])):
      return np.zeros(4, dtype=np.float64)
  ```
* **Offline Verification:**
  Run `scripts/audit_deployment_logic.py` check #6: verify non-finite input returns clean zero velocity vector.

---

### [ ] Item 1.2: Differentiate Between Unknown Altitude and Low Altitude
* **Locations:**
  * [`run_tracker.py:L624-660`](run_tracker.py#L624-L660) (`enforce_altitude_limits`)
* **Description:**
  1. `VehicleState.altitude_rel_m` defaults to `0.0`. Before takeoff or if altitude packets drop, `current_alt` is $0.0\text{ m}$.
  2. Because $0.0 < (\text{min\_alt} - 0.2) = 1.8\text{ m}$, `enforce_altitude_limits()` flags `FLOOR_BREACH` and commands **forced climb $-0.8\text{ m/s}$**, regardless of whether the vehicle is armed or airborne.
  3. If `current_alt` is `NaN`, comparisons return `False`, returning `(v_down, "OK")` and bypassing all ground and ceiling safety limits.
  4. The code must explicitly separate:
     - **Unknown Altitude (`None` / `NaN` / uninitialized):** Inhibit autonomous climb, clamp descent to 0, or refuse autonomous GUIDED engagement.
     - **Low Altitude (Valid reading $< 2.0\text{ m}$ while airborne):** Apply progressive floor cushion and climb away from ground.
* **Possible Fix:**
  In `run_tracker.py`:
  ```python
  # Explicitly distinguish unknown altitude from low altitude
  if current_alt is None or not math.isfinite(current_alt) or not self.vehicle_state.in_air:
      # If sitting on ground or altitude is unknown, clamp descent to 0 but NEVER force climb
      return max(0.0, v_down), "GROUND_OR_INVALID_ALT"
  ```
* **Offline Verification:**
  Run `enforce_altitude_limits(0.0, 0.0)` with `in_air=False`: ensure output is `0.0 m/s`, not `-0.8 m/s`.

---

### [ ] Item 1.3: Verify Telemetry Freshness & Latch Stale-Stream Failsafe
* **Locations:**
  * [`run_tracker.py:L130-170`](run_tracker.py#L130-L170) (`VehicleState`)
  * [`control/autopilot_bridge.py:L380-450`](control/autopilot_bridge.py#L380-L450) (`telemetry_worker`)
* **Description:**
  `VehicleState` stores `pitch_rad`, `roll_rad`, `yaw_deg`, and `altitude_rel_m` with no freshness timestamps. While `autopilot_bridge.py` monitors overall vehicle heartbeat (with a 3.0-second timeout), individual high-rate telemetry streams (ATTITUDE at 50 Hz, GLOBAL_POSITION_INT at 20 Hz) can stall independently. If the telemetry thread freezes or drops packets, the control loop continues running at 50 Hz using frozen attitude/altitude from seconds ago.
* **Possible Fix:**
  Add a `last_telemetry_time` timestamp to `VehicleState` and verify freshness before computing commands:
  ```python
  telemetry_age = time.perf_counter() - self.vehicle_state.last_telemetry_time
  if telemetry_age > 0.35: # More than 350ms without fresh attitude/altitude
      # Declare telemetry stale and command emergency level holding hover
      self.autopilot.send_hold_hover()
      return
  ```

---

### [ ] Item 1.4: Stale Distance Sensor Latching & Missing Orientation Filter
* **Locations:**
  * [`run_tracker.py:L624`](run_tracker.py#L624) (`get_current_altitude`)
  * [`control/autopilot_bridge.py:L436-443`](control/autopilot_bridge.py#L436-L443) (`DISTANCE_SENSOR` parser)
* **Description:**
  1. `distance_sensor_m` has no timestamp. If a downward lidar disconnects or freezes mid-flight, the last reading (e.g. `0.45 m`) stays cached forever.
  2. `get_current_altitude()` unconditionally prefers `distance_sensor_m` over `altitude_rel_m`. If stuck at `0.45 m`, the drone continuously believes it is below floor limit and climbs indefinitely.
  3. `autopilot_bridge.py` does not check `msg.orientation == 25` (`MAV_SENSOR_ROT_PITCH_270` for downward-facing). Any horizontal obstacle rangefinder will overwrite altitude!
* **Possible Fix:**
  In `autopilot_bridge.py`:
  ```python
  elif msg_type == "DISTANCE_SENSOR":
      # 25 is MAV_SENSOR_ROT_PITCH_270 (downward-facing AGL rangefinder)
      if getattr(msg, "orientation", 25) == 25:
          dist_m = float(msg.current_distance) / 100.0
          with self.lock:
              self.distance_sensor_m = dist_m
              self.last_distance_time = time.perf_counter()
  ```
  In `run_tracker.py`:
  ```python
  # Only use distance sensor if received within the last 0.35 seconds
  if (self.vehicle_state.distance_sensor_m is not None and 
      (time.perf_counter() - getattr(self.vehicle_state, "last_distance_time", 0)) < 0.35 and
      0.05 < self.vehicle_state.distance_sensor_m < 50.0):
      return self.vehicle_state.distance_sensor_m
  ```

---

### [ ] Item 1.5: Frame Exposure Timestamping (Old Frames Looking "Too Fresh")
* **Locations:**
  * [`run_tracker.py:L996-1066`](run_tracker.py#L996-L1066) (`vision_worker`)
* **Description:**
  In `vision_worker()`, `ret, frame = cap.read()` acquires the frame at line 996. Then preprocessing, adaptive shadows, and YOLO inference run (taking 25–40 ms). Only **after inference finishes** at line 1066 is `self.perception_state.last_update_time = time.perf_counter()` assigned.
  In `control_loop()`, `time_since_vision = now - last_update_time` measures time relative to inference completion, not optical capture. If the camera driver buffers frames for 100 ms or inference lags, a 140 ms old image appears "0 ms fresh" to the watchdog.
* **Possible Fix:**
  Timestamp the frame immediately when `cap.read()` returns and pass `frame_capture_time` to `PerceptionState`:
  ```python
  ret, frame = cap.read()
  capture_time = time.perf_counter()
  ...
  # Store capture timestamp alongside telemetry:
  self.perception_state.last_capture_time = capture_time
  # Watchdog evaluates: now - self.perception_state.last_capture_time
  ```

---

### [ ] Item 1.6: Verify Progressive Deceleration in Search Mode
* **Locations:**
  * [`control/pid_controller.py:L224-248`](control/pid_controller.py#L224-L248) (`compute_cmd`)
  * [`run_tracker.py:L1147-1193`](run_tracker.py#L1147-L1193) (`control_loop`)
* **Verification & Description:**
  - **In `search_mode == "brake"` (Current Default):** `compute_cmd()` immediately commands `[0.0, 0.0, 0.0, 0.0]`. Deceleration is governed by the outer slew limiter in `run_tracker.py` line 1192: `cmd_safe = np.clip(raw_cmd, prev_cmd - max_accel * dt)`. It does decelerate smoothly down to 0 at $4.0\text{ m/s}^2$ ($0.75\text{ s}$ from $3.0\text{ m/s}$).
  - **In `search_mode == "legacy_coast"`:** Line 242 computes `cmd_vx = max(0.0, self.prev_cmd_vx - 1.5 * dt)` but **never updates `self.prev_cmd_vx`**! As reproduced in the audit script, it outputs a constant `2.97 m/s` for an entire second without decelerating.
* **Possible Fix:**
  Ensure `search_mode = "brake"` remains the strictly enforced default, and if the legacy ramp is used, update `self.prev_cmd_vx`:
  ```python
  if self.lost_time < 1.0:
      self.prev_cmd_vx = max(0.0, self.prev_cmd_vx - 1.5 * dt)
      return np.array([self.prev_cmd_vx, 0.0, 0.0, 0.0], dtype=np.float64)
  ```

---

## 🟡 Tier 2: Tracking & Estimation Discrepancies

### [ ] Item 2.1: 8D Kalman State Transition Assuming Fixed 30 FPS
* **Locations:**
  * [`perception/pipeline.py:L47-63`](perception/pipeline.py#L47-L63) (`KalmanBoxTracker`)
* **Description:**
  The transition matrix $F$ is initialized once with hardcoded $dt = 1/30$. When running CSI Mode 1 (60 FPS, $dt = 16.7\text{ ms}$) or TensorRT inference (38 FPS, $dt = 26\text{ ms}$), `predict()` moves state by the wrong physical displacement, skewing velocity estimation and gating.
* **Possible Fix:**
  Allow dynamic `dt` in `KalmanBoxTracker.predict(dt=None)`:
  ```python
  def predict(self, dt: Optional[float] = None, ego_shift=None):
      eff_dt = float(np.clip(dt, 0.005, 0.1)) if dt is not None else self.dt
      # Update velocity transition elements dynamically
      F = self.F.copy()
      for i in range(4):
          F[i, i + 4] = eff_dt
      self.x = np.dot(F, self.x)
      self.P = np.dot(np.dot(F, self.P), F.T) + self.Q * (eff_dt / self.dt)
  ```

---

### [ ] Item 2.2: Target Re-Acquisition Selecting Coasting / Phantom Track
* **Locations:**
  * [`perception/pipeline.py:L333-339`](perception/pipeline.py#L333-L339) (`_update_primary_lock`)
* **Description:**
  When acquiring a new primary target (`primary_track_id is None`), the tracker selects the confirmed track with the most hits. It does not check `is_coasting`. If a track is dead-reckoning in the dark without a visual detection, it can become the primary lock, commanding flight toward an unverified prediction.
* **Possible Fix:**
  In `DroneTracker._update_primary_lock()`:
  ```python
  # Only lock onto confirmed tracks that have an active detection in the current frame
  visible_confirmed = [t for t in active_tracks if t["is_confirmed"] and not t["is_coasting"]]
  if self.primary_track_id is None and len(visible_confirmed) > 0:
      best_trk = max(visible_confirmed, key=lambda x: (x["hits"], x["confidence"]))
      self.primary_track_id = best_trk["id"]
  ```

---

### [ ] Item 2.3: Production Model Weights List Missing Latest Calibrated Models
* **Locations:**
  * [`run_tracker.py:L1525-1535`](run_tracker.py#L1525-L1535) (`candidate_weights`)
* **Description:**
  `candidate_weights` defaults to `yolov8n_drone_768_ciou_best.engine/.pt`. The newly trained **Hybrid NWD model** (`yolov8n_drone_768_hybrid_nwd_c51_best.pt`) and local **fine-tuned model** (`yolov8n_nougat_pede_768_best.pt`) are omitted. Running without `--weights` silently reverts to the older pure-CIoU baseline.
* **Possible Fix:**
  Prepend latest weights to `candidate_weights`:
  ```python
  candidate_weights = [
      DETECTION_DIR / "weights/yolov8n_nougat_pede_768_best.engine",
      DETECTION_DIR / "weights/yolov8n_nougat_pede_768_best.pt",
      DETECTION_DIR / "weights/yolov8n_drone_768_hybrid_nwd_c51_best.engine",
      DETECTION_DIR / "weights/yolov8n_drone_768_hybrid_nwd_c51_best.pt",
      DETECTION_DIR / "weights/yolov8n_drone_768_ciou_best.engine",
      DETECTION_DIR / "weights/yolov8n_drone_768_ciou_best.pt",
  ]
  ```

---

### [ ] Item 2.4: Camera Intrinsics & Lens Tilt Override Mismatch
* **Locations:**
  * [`run_tracker.py:L223-283`](run_tracker.py#L223-L283) vs [`control/spatial.py:L21-24`](control/spatial.py#L21-L24)
* **Description:**
  CLI arguments `--uptilt`, `--hfov`, and `--vfov` configure the controller, but `compute_ego_motion_compensation` in `spatial.py` uses hardcoded default arguments ($15^\circ$ uptilt, $60^\circ$ HFOV, $45^\circ$ VFOV). If lens parameters are overridden via CLI, the Kalman filter and visual servoing decouple across different geometries.
* **Possible Fix:**
  Explicitly pass `camera_uptilt_deg`, `hfov_deg`, and `vfov_deg` from `run_tracker.py` into `DroneTrackingPipeline` and `compute_ego_motion_compensation`.

---

## 🟢 Tier 3: System Performance & Polish

### [ ] Item 3.1: Decouple Video Disk I/O from 50 Hz MAVLink Control Loop
* **Locations:**
  * [`run_tracker.py:L1122-1250`](run_tracker.py#L1122-L1250) (`display_loop` vs `control_loop`)
* **Description:**
  Writing video frames to disk via OpenCV `VideoWriter.write()` and GUI rendering run on the same `asyncio` event loop as the MAVLink control loop. On the Jetson Orin Nano, micro-SD / eMMC write stalls (20–40 ms) cause jitter in the 50 Hz heartbeat.
* **Possible Fix:**
  Offload disk writing to a background worker thread using `queue.Queue`:
  ```python
  self.video_write_queue = queue.Queue(maxsize=60)
  # Background worker pops frames and writes to disk without blocking asyncio
  ```

---

### [ ] Item 3.2: Clarify Status of 35 ms Camera Latency & Ego-Motion Compensation
* **Locations:**
  * [`run_tracker.py:L1015-1017`](run_tracker.py#L1015-L1017) (`vision_worker`)
* **Verification & Description:**
  In `vision_worker()`, `target_att_time = now - 0.035` looks back 35 ms in the attitude ring buffer. This was strictly designed for the experimental ego-motion compensation module in `pipeline.process_frame()`. In `control_loop()` (line 1153), the visual servoing controller uses `current_pitch = self.vehicle_state.pitch_rad` (instantaneous attitude).
  If you have chosen to disable or bypass complex ego-motion compensation in the Kalman filter for flight trials, this 35 ms lookup is completely unused by the flight controller.
* **Action:**
  Document in code and thesis whether ego-motion compensation is active or bypassed for the flight test.

---

### [-] Item 3.3: Target Identity Switching (Deprioritized for Current Flight ATP)
* **Locations:**
  * [`perception/pipeline.py:L250-275`](perception/pipeline.py#L250-L275) (`Hungarian matching`)
* **Note:**
  In single-target flight pursuit against one known target drone in an open sky field, multi-target ID switching is a negligible risk compared to physical flight safety. Complex appearance embeddings / ReID are deprioritized for this milestone. Basic gating (Item 2.2) is sufficient.

---

## Step-by-Step Verification Protocol (Props-Off)

1. **Benchtop Software Run:** Execute `python scripts/audit_deployment_logic.py` and verify all assertions pass once remediated.
2. **SITL / Drone Bench Test (Props Removed):**
   * Connect Jetson to Pixhawk via UART (`/dev/ttyTHS0:921600`).
   * Boot in `STABILIZE` mode; verify companion computer does not attempt to send velocity setpoints.
   * Switch transmitter to `GUIDED` mode; verify clean transition, smooth 50 Hz telemetry, and zero command jitter.
   * Disconnect or cover camera lens; verify watchdog enters immediate zero-velocity holding hover within $0.40\text{ s}$.
   * Flip transmitter switch back to `LOITER` / `ALT_HOLD`; verify immediate pilot manual takeover.
