# Deployment pipeline audit — 16 October 2026

## Assessment and scope

**Recommendation: do not enable autonomous flight with the current bridge.** Confirmed faults exist in failure handling, telemetry validity, and command validation. Resolving these does not by itself demonstrate physical flight safety.

Reviewed working tree based on commit `efe9d67`: `run_tracker.py`, `perception/pipeline.py`, `control/pid_controller.py`, `control/spatial.py`, simulation, learned-controller interfaces, representative benchmark code, reports, and deployment roadmap. Existing user changes in `perception/train.py` and untracked `perception/preparation/compute_nwd_constant.py` were left untouched. No production code was changed, hardware opened, model executed, or commands transmitted.

Evidence categories: **confirmed** means directly visible in the execution path or reproduced offline; **validation gap** means the repository cannot establish deployed behavior. Priorities below concern deployment impact rather than likelihood of a crash in every run.

## Deployment blockers

### 1. Vision watchdog activates pursuit recovery instead of a failure hold — critical, confirmed

Locations: `run_tracker.py:1127`, `control/pid_controller.py:221`.

Expired perception only changes status to `SEARCHING`. That controller branch coasts forward for roughly one second, pauses, then commands a continuous 15 degrees/s yaw scan. It is used for camera failure and inference failure as well as ordinary target loss. Even initial acquisition eventually starts a scan. The outer slew limiter does not make this a zero-command watchdog.

Offline reproduction using the actual `control_loop` with no camera updates produced a positive forward command. A separate controller reproduction produced the persistent yaw scan. Freshly processed blank images have a different failure path: they can continue `COASTING` first, so the watchdog timeout is not a maximum blind-flight duration.

Required correction: separate sensor/telemetry failure, valid target loss, acquisition, and authorized tracking states. A fault must select an explicitly defined stop/hold or autopilot fallback independently of target-search logic. Validate the physical braking behavior; a zero velocity request does not stop a moving airframe instantly.

### 2. Shutdown sends the stop command after potentially slow cleanup — critical, confirmed

Locations: `run_tracker.py:1374–1425`.

Control is cancelled before a thread join of up to 1.5 seconds, video finalization, optional ablation plotting, and HTTP shutdown. Only afterward is a zero setpoint sent. Cleanup exceptions can skip that send. There is no confirmed handover to a holding mode or offboard stop. The vision thread can also outlive the join and still touch resources being closed.

MAVSDK automatically resends its last setpoint independently of the Python control iteration. Therefore, cancelling the coroutine cannot be assumed to expire the last velocity command. Actual server/process lifecycle behavior must be tested with the installed version.

Required correction: prioritize and bound the flight-state transition before nonessential cleanup; explicitly verify ownership release and fallback. Keep a fallback in the flight controller for companion power loss and a surviving-but-stale MAVSDK server.

### 3. Telemetry loss does not inhibit flight commands — critical, confirmed

Locations: `run_tracker.py:129–186`, `504–603`, `1133–1195`, `1313`.

Attitude, position, range, and velocity have no freshness/validity contract. Listener failures print messages and leave old values available. Connection state is only observed until the first successful connection; `is_connected` is never subsequently cleared. `watch_flight_mode` discards each received mode instead of updating state. Armed and battery fields remain defaults. No current estimator-health or in-air gate protects dispatch, and telemetry tasks are not supervised alongside control/display.

Consequences include continued tracking with frozen attitude or altitude and no reliable application-level indication of pilot takeover. A later re-entry into a command-accepting mode can expose stale controller state.

Required correction: timestamp and validate required streams, supervise tasks and connection transitions, and latch an abort on loss of authority or required telemetry. Re-engagement needs an explicit transition and controller-state reset. Battery and geofence protection should also be independently configured in the autopilot; this review cannot establish their settings.

### 4. Missing or invalid altitude can trigger unintended climb or bypass protection — critical, confirmed

Locations: `run_tracker.py:606–678`, `1170`.

Before position arrives, relative altitude defaults to zero. Floor protection interprets it as a real breach and replaces even a zero command with a **0.8 m/s climb**. This applies without checking that the vehicle is airborne or autonomous control is authorized.

Any stored range reading between 0.05 and 50 m wins indefinitely, without sample age, sensor orientation, reported limits, or quality checks. A stale 0.5 m reading can override a valid 10 m relative altitude. Conversely, relative altitude above home is not necessarily height above terrain. Slant range under tilt also needs a defined interpretation.

With NaN altitude all comparison branches are skipped and the original descent command is returned with status `OK`; this was reproduced.

Required correction: represent unknown altitude explicitly and inhibit autonomous engagement until its source and validity are established. Do not treat missing measurements as a real floor breach. Separate range-derived AGL and home-relative altitude limits.

### 5. NaN/Inf inputs have no final command barrier — critical, confirmed

Locations: `control/pid_controller.py:237–444`, `run_tracker.py:1148–1191`.

There is no final finiteness check on the entire four-axis command or its required inputs. `np.clip` and slew limiting do not sanitize NaN. A locked detection with NaN horizontal error produced NaN vertical and yaw outputs offline. A poisoned integral or previous-command state can persist after the original bad sample disappears.

Required correction: validate measurements and the final command vector, reject unknown statuses, reset poisoned state, and enter a defined fault transition before serialization. Do not rely on MAVSDK/autopilot rejection behavior as the only barrier.

### 6. Autopilot ownership and mode integration are incomplete — high, confirmed gap

Locations: `run_tracker.py:1182–1195`, `1300–1328`; project overview versus Week 5 report.

The bridge sends setpoints but never calls `offboard.start()` or verifies an externally established control mode. It neither implements nor documents a complete ownership handshake. Setpoint setters are not proof that the vehicle accepted or executed a command; catching `OffboardError` is not a reliable pilot-takeover detector.

Older project documents describe PX4; physical logs describe ArduPilot. MAVSDK's documented Offboard workflow uses PX4 Offboard, whereas ArduPilot documents Guided-mode command handling. This is not evidence that all MAVSDK communication with ArduPilot fails. It is evidence that mode entry, setpoint acceptance, manual override, timeouts, and re-entry must be established for the exact firmware/API combination.

Required validation: record exact autopilot firmware and MAVSDK/server versions, reproduce the full handover and abort sequence in matching SITL, then verify props-off. Do not simply add automatic arming or mode entry to make the existing bridge move.

## Target loss and perception logic

### 7. Search coasting does not progressively decelerate — high, confirmed

Location: `control/pid_controller.py:225`.

The early-return branch calculates `prev_cmd_vx - 1.5*dt` but never updates `prev_cmd_vx` or the normal `prev_cmd` state. Starting at 3 m/s, 40 successive 20 ms calls all return **2.97 m/s**, rather than a decaying ramp. The outer limiter follows this nearly unchanged reference until the next phase. Reacquisition also retains pre-loss integrator, error, and command history.

### 8. COASTING uses full active guidance, including acceleration — high, confirmed

Locations: `perception/pipeline.py:289–315`, `790`; `control/pid_controller.py:221–235`.

Only exactly `SEARCHING` gets special treatment. A predicted box marked `COASTING` uses the same feedback as a measured lock and resets the lost timer. A synthetic coasting track produced positive forward/lateral acceleration. The default 15 missed-frame allowance is roughly 0.5 seconds at 30 FPS but one second at 15 FPS, before the additional search coast.

Required correction: use elapsed time since the last associated real detection and an uncertainty bound. Define whether blind recovery is permitted and prevent predictions alone from renewing that authorization.

### 9. Target selection can silently change what the drone follows — high, confirmed

Locations: `perception/pipeline.py:257–273`, `324–339`, `642–666`.

Association can accept a detection within an 80-pixel minimum fallback gate even without sufficient IoU. Primary selection favors lifetime hits then confidence; no target identity or operator-selected authorization is present. Selection can acquire an already coasting confirmed track. Two lifetime hits need not be consecutive. After expiry another confirmed object can become primary automatically.

Required correction: define target eligibility, re-acquisition, identity-switch handling, and confidence/uncertainty gates. Test crossing targets, birds, false positives, and intermittent detections. A detector confidence threshold alone does not solve identity.

### 10. Weight fallback can silently load a different detector — high, confirmed

Locations: `run_tracker.py:246–264`; `perception/pipeline.py:402–419`, `656–666`.

An explicitly requested PyTorch model without a same-name engine can be replaced by a different production engine found on disk. Missing pipeline weights can fall back as far as generic `yolov8n.pt`. Detection class IDs are not filtered before tracking. In that generic fallback case, ordinary COCO detections can enter drone guidance.

Required correction: flight configuration must select an exact validated model/class schema and fail closed on missing or incompatible artifacts. Log its checksum and engine/input configuration. GPU compatibility and inference accuracy must be checked on the actual Jetson.

## Geometry, time, and control consistency

### 11. Camera calibration overrides do not reach ego compensation — high, confirmed

Locations: `run_tracker.py:223–283`; `perception/pipeline.py:514–548`; defaults in `control/spatial.py`.

The controller receives `--uptilt`, `--hfov`, and `--vfov`, but ego-motion calculations use hard-coded function defaults of 15/60/45 degrees. Changing physical calibration thus changes one side of the pipeline only. Image dimensions are propagated, but intrinsics/mount geometry are not.

The 1280x720 stream and 60x45-degree defaults also need physical calibration. With square pixels and no special image resampling, 60-degree horizontal FOV at 16:9 implies about 36 degrees vertical, not 45. Actual sensor crops and resizing determine the correct values; this is a calibration warning, not proof of your lens's FOV.

A numerical combined roll/pitch/yaw projection/unprojection round trip agreed to approximately 1e-17 radians. The basic simulator/controller transform pair is internally consistent for that case. This does not validate camera mounting, lens distortion, telemetry convention, or timing on hardware.

### 12. Frame age is replaced by inference-completion freshness — high, confirmed

Locations: `run_tracker.py:989–997`, `1046`, `1123`, `1133–1143`.

Watchdog time starts after inference finishes, not at sensor exposure. A slow or buffered frame is briefly marked fresh regardless of how old its pixels are. Attitude history is indexed using host receipt timestamps and a fixed 35 ms offset, not a synchronized camera/autopilot acquisition clock. The controller then combines the old image errors with current attitude instead of the image-aligned attitude used by the tracker.

Required correction: carry acquisition timestamp/frame ID through processing and enforce maximum observation age. Establish the camera and telemetry timing experimentally; distinguish image compensation from command prediction.

The CSI appsink string omits an explicit `max-buffers`, but **this audit does not claim an unlimited queue is proven**: upstream OpenCV's GStreamer backend sets a one-buffer limit when absent. Verify the deployed OpenCV backend and measure frame age under overload rather than inferring queue behavior from the pipeline string alone.

### 13. Kalman prediction uses fixed 30 FPS time — medium/high, confirmed

Locations: `perception/pipeline.py:57–62`, `107–118`, `211`, `282`.

The transition matrix is initialized once with 1/30 s and never follows actual camera intervals. The measured `ego_dt` only affects compensation. At 15 FPS, 15 predictions advance a 60 px/s state by 30 px despite one second of elapsed time. Velocity estimates, gating, missed-frame horizons, and covariance behavior therefore depend on throughput. Correct this together with time-scaled process noise, not just one matrix entry.

### 14. Controller derivatives use control-loop time on asynchronously sampled measurements — medium/high, confirmed

Locations: `run_tracker.py:1114–1143`; `control/pid_controller.py:342`.

At a 50 Hz control rate and approximately 30 Hz perception rate, some iterations repeat a measurement while others differentiate an image update over a 20 ms control interval. Derivative impulses and filtering depend on scheduling. Add frame identity and measurement timing; define which updates change the measurement derivative and integrator. Double slew limiting also exists in controller and bridge and needs to be included in physical response tests.

### 15. Altitude override and controller history disagree — medium/high, confirmed

Locations: `run_tracker.py:1164–1173`; `control/pid_controller.py:441`.

Previous-command state is saved before altitude protection replaces the vertical command. Internal integration/slew state therefore follows a command that may never reach the vehicle. Active floor/ceiling corrections can also bypass the bridge's vertical slew limit. Feed the actually applied constraint back into control state and define priority between braking, limits, and fault actions.

The altitude cushion is based on position alone, not measured descent speed or calibrated vertical lag. It is a command limiter, not proof that the aircraft can stop before crossing the floor. The same distinction applies to forward standoff control and box-size-based braking.

### 16. Pixel size is not a guaranteed separation measurement — validation gap

Locations: `perception/pipeline.py:778–797`; `control/pid_controller.py:371–399`.

The largest box dimension is treated as an inverse-distance proxy. Target yaw, aspect ratio, rotor visibility, partial occlusion, clipping, image resolution, and detector jitter all alter it without a corresponding range change. `--standoff-dist` alone does not recalibrate `--target-size`. A 63 px setpoint needs validation for the actual target and camera mode; it cannot independently guarantee 3.5 m clearance.

No obstacle detection or horizontal geofence is implemented in the bridge. Valid target tracking can still command flight into an obstacle. Any autopilot geofence and operational restrictions are external and were not inspected.

## Misleading deployment diagnostics and evidence

### 17. Telemetry dry-run overwrites real altitude — medium, confirmed

Locations: `run_tracker.py:373`, `1176–1179`.

`--dry-run --connection ...` starts real position telemetry but also integrates simulated altitude into the same field every control iteration. The displayed altitude and safety logic can oscillate between measured and simulated values, undermining bench conclusions. Keep simulation state separate and simulate only when no real telemetry source is selected.

### 18. Main control scheduling shares blocking display/recording work — high validation gap

Locations: `run_tracker.py:1204–1298`, `1184`.

HUD rendering, JPEG compression, video writes, console output, and GUI operations execute synchronously on the same asyncio thread as control and telemetry. A slow filesystem or encoder can stall both the control watchdog and telemetry consumers while MAVSDK continues resending an old command. The send RPC also has no explicit bounded wait. A separate vision thread does not isolate these failure modes.

Required validation: instrument real control deadlines, RPC latency, observation age, and telemetry age with recording/streaming enabled, then inject disk/encoder/RPC stalls. Move noncritical blocking work out of the control scheduler and give watchdog enforcement an independent execution path.

### 19. Translational compensation option does not enable production compensation — medium, confirmed

Locations: `run_tracker.py:1565`; `perception/pipeline.py:514–548`.

`--enable-trans-ego` is parsed but never consumed. Primary tracking uses rotation-only compensation. Translation is applied only to the parallel full-compensation ablation tracker, which does not drive commands. That ablation uses a fixed depth and yaw-only NED-to-heading velocity transformation; it is not a fully attitude-rotated body velocity. Documentation describing production 6-DOF compensation overstates the execution path.

### 20. Exceptions can silently degrade compensation or leave an unhealthy worker alive — medium/high, confirmed

Locations: `run_tracker.py:1033–1040`, `958–1100`; `perception/pipeline.py:22–28`.

A broad `TypeError` retry can hide an internal perception error by calling the stateful pipeline again without ego telemetry. Other vision-thread exceptions have no general error propagation/finally cleanup; a dead worker can leave control running its faulty watchdog search. Spatial-import failure silently disables compensation. Flight mode should treat unexpected processing errors as explicit faults rather than compatibility fallbacks.

### 21. Latency and simulation reports are not deployment acceptance evidence — confirmed limitation

Locations: `benchmarks/measure_jetson_latency.py:117–186`; `outputs/benchmarks/jetson_orin_nano_latency_report.md`; `control/simulation.py:247–300`.

The latency profiler processes synthetic memory images and sums software durations. It excludes physical exposure/ISP age, live capture stalls, command transport and application, and aircraft response. Synthetic imagery is not guaranteed to create an active drone track, so a controller timing can exercise search/fallback instead of the intended locked path. The saved report contains only one sample and inconsistent explanatory totals. The Week 5 report describes a different 200-frame run; its result is reported evidence, not reproduced in this audit.

Simulator camera telemetry can mark an out-of-view target `COASTING` while including ground-truth distance and projected direction; this differs from the production tracker and its eventual expiry. Benchmarks do not execute the actual bridge's complete lifecycle, telemetry validity, altitude constraints, and scheduling. Existing test scripts mostly cover simulation/plots, not these deployment faults.

The learned policy is not selectable in the current flight bridge: it always constructs `KinematicVisualServoController`. A future integration must reconcile trained standoff, image normalization, observation dimensions, measured velocities/yaw rate, update rate, and recurrent resets. It is not currently a validated drop-in flight option.

## Offline verification performed

Run `.venv/bin/python scripts/audit_deployment_logic.py` from the repository root.

All **10 defect-characterization checks reproduced**: constant search-coast output; fault-induced scan; acceleration during coasting; nonfinite command propagation; missing-altitude climb; unqualified range preference; NaN-altitude bypass; fixed Kalman timestep; primary selection of a coasting track; and forward output from the actual runtime watchdog loop. Tests execute selected source definitions through AST, avoiding model imports, GUI, cameras, and flight connections. They intentionally assert current faulty behavior and must be rewritten as regression expectations when fixes land. They are not an end-to-end integration test or flight qualification.

An additional offline simulator/controller projection round-trip check passed for one combined-attitude case. No hardware or SITL tests were run.

## Required sequence before considering flight

1. Resolve fault-state handling, command validity, telemetry freshness, authority tracking, and priority shutdown. Add regression tests using the real bridge and mocked autopilot/capture failures.
2. Establish one exact firmware/API deployment profile. Test mode entry, manual override, re-entry, loss of connection, companion crash, and surviving-server stale setpoints in matching SITL.
3. Freeze and verify camera calibration, selected model checksum/class schema, image size, target-size calibration, and timestamp behavior on the Jetson. Measure actual frame age and deadline tails under full recording/streaming load.
4. Perform props-off integration tests with the real flight controller. Verify command signs, sensor disconnection, invalid/stale measurements, and fallback behavior from recorded autopilot state, not just the HUD.
5. Verify aircraft dynamics and stopping response with appropriately supervised incremental tests before target-following trials. Keep manual takeover and autopilot-native limits independent of this process. Physical motor repair, power integrity, vibration, and failsafe configuration remain unverified here.

## Primary references checked

- MAVSDK Offboard workflow and automatic setpoint resend: https://mavsdk.mavlink.io/main/en/cpp/guide/offboard.html
- ArduPilot Guided command handling and velocity timeout: https://ardupilot.org/dev/docs/copter-commands-in-guided-mode.html
- ArduPilot Guided timeout parameter: https://docs.ardupilot.org/copter/docs/ac2_guidedmode.html
- MAVLink frame definitions: https://mavlink.io/en/messages/common.html
- GStreamer appsink queue properties: https://gstreamer.freedesktop.org/documentation/app/appsink.html
- OpenCV GStreamer sink initialization: https://github.com/opencv/opencv/blob/4.x/modules/videoio/src/cap_gstreamer.cpp

These describe upstream behavior; installed firmware/library versions still need direct validation.
