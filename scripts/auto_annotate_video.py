import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from ultralytics import YOLO

def main():
    video_path = './perception/raw_data/own-collected-data/My Movie 4.mp4'
    gt_xml_path = os.path.expanduser('~/Downloads/annotations.xml')
    model_path = 'perception/weights/yolov8n_drone_768_ciou_best.pt'
    output_xml_path = './perception/raw_data/own-collected-data/annotations_auto.xml'
    output_video_path = './perception/raw_data/own-collected-data/My_Movie_4_annotated_preview.mp4'
    dataset_dir = './perception/datasets/my_movie_annotated'

    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")

    # 1. Load Human Ground Truth from CVAT XML for frames 98-195
    gt_boxes = {}
    if os.path.exists(gt_xml_path):
        tree = ET.parse(gt_xml_path)
        for b in tree.findall('.//box'):
            if b.attrib.get('keyframe') == '1':
                f = int(b.attrib['frame'])
                if 98 <= f <= 195:
                    box = [float(b.attrib['xtl']), float(b.attrib['ytl']),
                           float(b.attrib['xbr']), float(b.attrib['ybr'])]
                    gt_boxes[f] = box
        print(f"Loaded {len(gt_boxes)} human ground-truth keyframes from CVAT (frames 98-195).")

    # 2. Run YOLO on all frames
    print(f"Loading detector: {model_path}")
    model = YOLO(model_path)
    
    # Store raw detections per frame: list of [x1, y1, x2, y2, conf]
    raw_detections = {}
    print("Running YOLO inference across all frames...")
    
    batch_size = 16
    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while frame_idx < total_frames:
        frames_batch = []
        indices_batch = []
        for _ in range(batch_size):
            if frame_idx >= total_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frames_batch.append(frame)
            indices_batch.append(frame_idx)
            frame_idx += 1
            
        if not frames_batch:
            break
            
        results = model.predict(frames_batch, imgsz=768, conf=0.18, device='mps', verbose=False)
        for idx, res in zip(indices_batch, results):
            boxes = res.boxes
            if len(boxes) > 0:
                det_list = []
                for b in boxes:
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                    conf = float(b.conf[0].cpu().numpy())
                    w = xyxy[2] - xyxy[0]
                    h = xyxy[3] - xyxy[1]
                    # Filter out unreasonable aspect ratios or full-frame blobs
                    if 15 < w < 350 and 10 < h < 250 and 0.2 < (w/h) < 5.0:
                        det_list.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf])
                if det_list:
                    raw_detections[idx] = det_list

        if frame_idx % 200 == 0 or frame_idx >= total_frames:
            print(f"Processed {frame_idx}/{total_frames} frames...")

    print(f"Inference complete. Detections found in {len(raw_detections)} frames.")

    # 3. Track Linking & Smoothing
    # We construct continuous tracks. For Shot 1 (98-195), use human ground truth.
    # For other frames, link high-confidence detections and interpolate short dropouts (<= 6 frames).
    
    final_annotations = {} # frame_idx -> [x1, y1, x2, y2, source]
    
    # Inject human ground truth first
    for f, b in gt_boxes.items():
        final_annotations[f] = b + ['human_gt']

    # Identify detection tracks across the remaining frames
    active_track = None # {'last_frame': f, 'box': [x1,y1,x2,y2], 'history': [(f, box)]}
    all_tracks = []
    
    for f in range(total_frames):
        if f in gt_boxes:
            if active_track is not None:
                if len(active_track['history']) >= 5:
                    all_tracks.append(active_track)
                active_track = None
            continue
            
        dets = raw_detections.get(f, [])
        matched = False
        
        if active_track is not None:
            last_f = active_track['last_frame']
            gap = f - last_f
            if gap <= 8: # Allow bridging up to ~0.25s dropouts
                last_box = active_track['box']
                # Find closest detection within distance threshold
                best_det = None
                best_dist = 120.0 * (1.0 + gap * 0.2)
                for d in dets:
                    cx = (d[0] + d[2]) / 2.0
                    cy = (d[1] + d[3]) / 2.0
                    lcx = (last_box[0] + last_box[2]) / 2.0
                    lcy = (last_box[1] + last_box[3]) / 2.0
                    dist = np.hypot(cx - lcx, cy - lcy)
                    if dist < best_dist:
                        best_dist = dist
                        best_det = d
                        
                if best_det is not None:
                    # Bridge gap by linear interpolation
                    if gap > 1:
                        for gf in range(last_f + 1, f):
                            alpha = (gf - last_f) / float(gap)
                            interp_box = [
                                last_box[0] * (1 - alpha) + best_det[0] * alpha,
                                last_box[1] * (1 - alpha) + best_det[1] * alpha,
                                last_box[2] * (1 - alpha) + best_det[2] * alpha,
                                last_box[3] * (1 - alpha) + best_det[3] * alpha,
                            ]
                            active_track['history'].append((gf, interp_box, 'interpolated'))
                    active_track['history'].append((f, best_det[:4], 'yolo'))
                    active_track['box'] = best_det[:4]
                    active_track['last_frame'] = f
                    matched = True
            else:
                # Track ended
                if len(active_track['history']) >= 5:
                    all_tracks.append(active_track)
                active_track = None

        if not matched:
            # Check if any detection has high confidence to start a new track
            high_conf_dets = [d for d in dets if d[4] >= 0.40]
            if high_conf_dets:
                best_start = max(high_conf_dets, key=lambda d: d[4])
                if active_track is not None:
                    if len(active_track['history']) >= 5:
                        all_tracks.append(active_track)
                active_track = {
                    'last_frame': f,
                    'box': best_start[:4],
                    'history': [(f, best_start[:4], 'yolo')]
                }

    if active_track is not None and len(active_track['history']) >= 5:
        all_tracks.append(active_track)

    print(f"Extracted {len(all_tracks)} verified drone tracks outside of Shot 1.")
    for i, t in enumerate(all_tracks):
        start_f = t['history'][0][0]
        end_f = t['history'][-1][0]
        print(f"  Track {i+1}: frames {start_f} -> {end_f} ({len(t['history'])} frames)")
        for hf, hbox, hsrc in t['history']:
            final_annotations[hf] = hbox + [hsrc]

    print(f"Total annotated frames: {len(final_annotations)} (out of {total_frames}).")

    # 4. Generate CVAT Video 1.1 XML
    print(f"Writing CVAT XML to {output_xml_path}...")
    root = ET.Element('annotations')
    version = ET.SubElement(root, 'version')
    version.text = '1.1'
    meta = ET.SubElement(root, 'meta')
    task = ET.SubElement(meta, 'task')
    task_id = ET.SubElement(task, 'id')
    task_id.text = '1'
    task_name = ET.SubElement(task, 'name')
    task_name.text = 'My Movie 4'
    size = ET.SubElement(task, 'size')
    size.text = str(total_frames)

    # Human ground truth track
    track_gt = ET.SubElement(root, 'track', id='0', label='drone')
    for f in sorted(gt_boxes.keys()):
        b = gt_boxes[f]
        box_el = ET.SubElement(track_gt, 'box',
                               frame=str(f),
                               keyframe='1',
                               outside='0',
                               occluded='0',
                               xtl=f"{b[0]:.2f}",
                               ytl=f"{b[1]:.2f}",
                               xbr=f"{b[2]:.2f}",
                               ybr=f"{b[3]:.2f}",
                               z_order='0')

    # Auto-detected tracks
    for t_idx, t in enumerate(all_tracks):
        track_el = ET.SubElement(root, 'track', id=str(t_idx + 1), label='drone')
        for hf, hbox, hsrc in t['history']:
            box_el = ET.SubElement(track_el, 'box',
                                   frame=str(hf),
                                   keyframe='1' if hsrc == 'yolo' else '0',
                                   outside='0',
                                   occluded='0',
                                   xtl=f"{hbox[0]:.2f}",
                                   ytl=f"{hbox[1]:.2f}",
                                   xbr=f"{hbox[2]:.2f}",
                                   ybr=f"{hbox[3]:.2f}",
                                   z_order='0')

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
    print("CVAT XML exported successfully.")

    # 5. Export YOLO Dataset (Sampled every 3 frames for drone, plus 30 background frames)
    print(f"Creating YOLO formatted dataset in {dataset_dir}...")
    for split in ['train', 'val']:
        os.makedirs(f"{dataset_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{dataset_dir}/labels/{split}", exist_ok=True)

    # Sample annotated frames (every 3rd frame to avoid excessive redundancy)
    annotated_frames = sorted(final_annotations.keys())
    sampled_annotated = [f for i, f in enumerate(annotated_frames) if i % 3 == 0]
    
    # Negative frames (where drone is not present)
    negative_candidates = [f for f in range(0, total_frames, 20) if f not in final_annotations]
    sampled_negatives = negative_candidates[:40]

    all_dataset_samples = [(f, True) for f in sampled_annotated] + [(f, False) for f in sampled_negatives]
    np.random.seed(42)
    np.random.shuffle(all_dataset_samples)

    val_split_idx = int(len(all_dataset_samples) * 0.2)
    val_frames = set([f for f, _ in all_dataset_samples[:val_split_idx]])

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    written_count = 0
    for f, has_drone in all_dataset_samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret: continue
        split = 'val' if f in val_frames else 'train'
        img_name = f"my_movie_frame_{f:05d}.jpg"
        lbl_name = f"my_movie_frame_{f:05d}.txt"

        cv2.imwrite(f"{dataset_dir}/images/{split}/{img_name}", frame)
        with open(f"{dataset_dir}/labels/{split}/{lbl_name}", 'w') as lf:
            if has_drone:
                b = final_annotations[f]
                xc = ((b[0] + b[2]) / 2.0) / width
                yc = ((b[1] + b[3]) / 2.0) / height
                bw = (b[2] - b[0]) / width
                bh = (b[3] - b[1]) / height
                lf.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        written_count += 1

    with open(f"{dataset_dir}/data.yaml", 'w') as yf:
        yf.write(f"path: {os.path.abspath(dataset_dir)}\n")
        yf.write("train: images/train\n")
        yf.write("val: images/val\n")
        yf.write("names:\n  0: drone\n")

    print(f"YOLO dataset generated: {written_count} images ({len(sampled_annotated)} positive, {len(sampled_negatives)} background).")

    # 6. Render Preview Video (5x downsampled or full resolution)
    print(f"Rendering preview video with annotations to {output_video_path}...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vw = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for f in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        if f in final_annotations:
            b = final_annotations[f]
            src = b[4]
            color = (0, 255, 0) if src == 'human_gt' else ((0, 165, 255) if src == 'yolo' else (255, 200, 0))
            label_text = f"Drone ({src})"
            p1 = (int(b[0]), int(b[1]))
            p2 = (int(b[2]), int(b[3]))
            cv2.rectangle(frame, p1, p2, color, 2)
            cv2.putText(frame, label_text, (p1[0], max(20, p1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"Frame {f}/{total_frames}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out_vw.write(frame)
        if f % 300 == 0:
            print(f"Rendered {f}/{total_frames} preview frames...")

    cap.release()
    out_vw.release()
    print("Preview video rendered successfully!")

if __name__ == '__main__':
    main()
