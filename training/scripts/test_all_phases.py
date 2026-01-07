#!/usr/bin/env python3
"""Test visualization for all phases: Segmentation, Block Detection, and Text Line Detection."""

import sys
import os
os.chdir('/home/danny/Documents/personal/extension/training/comic-text-detector')
sys.path.insert(0, '/home/danny/Documents/personal/extension/training/comic-text-detector')

import torch
import cv2
import numpy as np
from pathlib import Path

from src.models import create_text_detector, TEXTDET_MASK, TEXTDET_DET
from src.models.heads import TEXTDET_BLOCK

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths to model checkpoints
    phase1_path = Path('runs/segmentation_v3/best.pt')
    phase2_path = Path('runs/detection_v5/best.pt')
    phase3_path = Path('runs/block_v5/best.pt')

    print(f"Loading Phase 1 (Segmentation) from: {phase1_path}")
    print(f"Loading Phase 2 (Detection) from: {phase2_path}")
    print(f"Loading Phase 3 (Block) from: {phase3_path}")

    # ========== Load checkpoints ==========
    phase1_ckpt = torch.load(phase1_path, map_location=device, weights_only=False)
    phase1_state = {k.replace('_orig_mod.', ''): v for k, v in phase1_ckpt['model_state_dict'].items()}

    phase2_ckpt = torch.load(phase2_path, map_location=device, weights_only=False)
    phase2_state = {k.replace('_orig_mod.', ''): v for k, v in phase2_ckpt['model_state_dict'].items()}

    phase3_ckpt = torch.load(phase3_path, map_location=device, weights_only=False)
    phase3_state = {k.replace('_orig_mod.', ''): v for k, v in phase3_ckpt['model_state_dict'].items()}

    # ========== Phase 1 Model (Segmentation) ==========
    model_seg = create_text_detector(
        backbone_name='yolo11s.pt',
        pretrained_backbone=False,
        freeze_backbone=True,
        device=str(device)
    )
    model_seg.load_state_dict(phase1_state, strict=False)
    model_seg.eval()
    model_seg.forward_mode = TEXTDET_MASK
    print("Phase 1 model loaded (segmentation)")

    # ========== Phase 2 Model (Detection/Text Lines) ==========
    model_det = create_text_detector(
        backbone_name='yolo11s.pt',
        pretrained_backbone=False,
        freeze_backbone=True,
        device=str(device)
    )
    model_det.load_state_dict(phase1_state, strict=False)
    model_det.initialize_db()

    # Load DBNet weights
    db_state = {k.replace('dbnet.', ''): v for k, v in phase2_state.items() if k.startswith('dbnet.')}
    if db_state:
        model_det.dbnet.load_state_dict(db_state)
    model_det = model_det.to(device)
    model_det.eval()
    model_det.forward_mode = TEXTDET_DET
    print("Phase 2 model loaded (detection)")

    # ========== Phase 3 Model (Block Detection) ==========
    model_blk = create_text_detector(
        backbone_name='yolo11s.pt',
        pretrained_backbone=False,
        freeze_backbone=True,
        device=str(device)
    )
    model_blk.load_state_dict(phase1_state, strict=False)
    model_blk.initialize_db()
    model_blk.initialize_block_detector()

    # Load block detector weights
    blk_state = {k.replace('block_det.', ''): v for k, v in phase3_state.items() if k.startswith('block_det.')}
    if blk_state:
        model_blk.block_det.load_state_dict(blk_state)
    model_blk = model_blk.to(device)
    model_blk.eval()
    model_blk.forward_mode = TEXTDET_BLOCK
    print("Phase 3 model loaded (block detection)")

    # ========== Find test images ==========
    test_images = list(Path('data/val/images').glob('*.jpg'))[:3]
    if not test_images:
        test_images = list(Path('data/train/images').glob('*.jpg'))[:3]

    if not test_images:
        print("No test images found!")
        return

    output_dir = Path('results/images')
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_idx, test_img_path in enumerate(test_images):
        print(f"\nProcessing: {test_img_path.name}")

        # Load and preprocess image
        img = cv2.imread(str(test_img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # Resize to model input size
        input_size = 1024
        img_resized = cv2.resize(img_rgb, (input_size, input_size))

        # Normalize
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # ========== Run inference ==========
        with torch.no_grad():
            # Phase 1: Segmentation
            seg_output = model_seg(img_tensor)
            seg_mask = seg_output.squeeze().cpu().numpy()

            # Phase 2: Detection (Text Lines)
            det_output = model_det(img_tensor)
            det_np = det_output.squeeze().cpu().numpy()
            if det_np.ndim == 3 and det_np.shape[0] == 2:
                prob_map = det_np[0]
                thresh_map = det_np[1]
            else:
                prob_map = det_np
                thresh_map = prob_map

            # Phase 3: Block Detection
            model_blk.block_det.training_mode = False
            blk_output = model_blk(img_tensor)

        # Process block detections
        blocks = []
        if blk_output is not None:
            blk_np = blk_output.squeeze().cpu().numpy()
            print(f"  Block output shape: {blk_np.shape}")

            # Block output format: [x, y, w, h, conf, cls] per detection
            # Filter by confidence threshold
            conf_thresh = 0.5
            nms_thresh = 0.4

            if blk_np.ndim == 2:
                # Collect all detections above threshold
                candidates = []
                for det in blk_np:
                    if len(det) >= 5:
                        x, y, w, h, conf = det[:5]
                        if conf > conf_thresh:
                            # Convert from normalized to pixel coords
                            x1 = int((x - w/2) * input_size)
                            y1 = int((y - h/2) * input_size)
                            x2 = int((x + w/2) * input_size)
                            y2 = int((y + h/2) * input_size)
                            # Clamp to image bounds
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(input_size, x2), min(input_size, y2)
                            if x2 > x1 and y2 > y1:
                                candidates.append([x1, y1, x2, y2, conf])

                # Apply NMS
                if candidates:
                    candidates = np.array(candidates)
                    boxes = candidates[:, :4]
                    scores = candidates[:, 4]

                    # NMS using OpenCV
                    indices = cv2.dnn.NMSBoxes(
                        boxes.tolist(),
                        scores.tolist(),
                        conf_thresh,
                        nms_thresh
                    )
                    if len(indices) > 0:
                        indices = indices.flatten()
                        for i in indices:
                            x1, y1, x2, y2, conf = candidates[i]
                            blocks.append((int(x1), int(y1), int(x2), int(y2), conf))

        print(f"  Segmentation mask range: {seg_mask.min():.3f} - {seg_mask.max():.3f}")
        print(f"  Detection prob_map range: {prob_map.min():.3f} - {prob_map.max():.3f}")
        print(f"  Detected {len(blocks)} text blocks")

        # ========== Create visualization ==========
        display_size = 512
        fig_h = display_size + 60
        fig_w = display_size * 4 + 50

        canvas = np.ones((fig_h, fig_w, 3), dtype=np.uint8) * 255

        # Prepare images
        img_display = cv2.resize(img_rgb, (display_size, display_size))

        # Segmentation mask visualization
        seg_mask_vis = (seg_mask * 255).astype(np.uint8)
        seg_mask_vis = cv2.resize(seg_mask_vis, (display_size, display_size))
        seg_mask_color = cv2.applyColorMap(seg_mask_vis, cv2.COLORMAP_JET)

        # Block detection visualization (original image with boxes)
        img_with_blocks = cv2.resize(img_rgb.copy(), (display_size, display_size))
        scale = display_size / input_size
        for (x1, y1, x2, y2, conf) in blocks:
            x1_s, y1_s = int(x1 * scale), int(y1 * scale)
            x2_s, y2_s = int(x2 * scale), int(y2 * scale)
            cv2.rectangle(img_with_blocks, (x1_s, y1_s), (x2_s, y2_s), (0, 255, 0), 2)
            cv2.putText(img_with_blocks, f'{conf:.2f}', (x1_s, y1_s - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Detection prob map visualization
        prob_vis = (prob_map * 255).astype(np.uint8)
        prob_vis = cv2.resize(prob_vis, (display_size, display_size))
        prob_color = cv2.applyColorMap(prob_vis, cv2.COLORMAP_JET)

        # Place images on canvas
        y_offset = 10
        x_positions = [10, display_size + 20, display_size * 2 + 30, display_size * 3 + 40]

        canvas[y_offset:y_offset+display_size, x_positions[0]:x_positions[0]+display_size] = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
        canvas[y_offset:y_offset+display_size, x_positions[1]:x_positions[1]+display_size] = seg_mask_color
        canvas[y_offset:y_offset+display_size, x_positions[2]:x_positions[2]+display_size] = cv2.cvtColor(img_with_blocks, cv2.COLOR_RGB2BGR)
        canvas[y_offset:y_offset+display_size, x_positions[3]:x_positions[3]+display_size] = prob_color

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_y = y_offset + display_size + 30
        cv2.putText(canvas, 'Original', (x_positions[0] + 200, label_y), font, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, 'Segmentation', (x_positions[1] + 180, label_y), font, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, f'Blocks ({len(blocks)})', (x_positions[2] + 190, label_y), font, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, 'Text Lines', (x_positions[3] + 190, label_y), font, 0.7, (0, 0, 0), 2)

        # Save
        output_path = output_dir / f'all_phases_test_{img_idx}.jpg'
        cv2.imwrite(str(output_path), canvas)
        print(f"  Saved to: {output_path}")

    print(f"\nAll visualizations saved to: {output_dir}")

if __name__ == '__main__':
    main()
