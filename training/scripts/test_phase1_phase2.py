#!/usr/bin/env python3
"""Test visualization for Phase 1 (segmentation) and Phase 2 (detection) models."""

import sys
import os
os.chdir('/home/danny/Documents/personal/extension/training/comic-text-detector')
sys.path.insert(0, '/home/danny/Documents/personal/extension/training/comic-text-detector')

import torch
import cv2
import numpy as np
from pathlib import Path

from src.models import create_text_detector, TEXTDET_MASK, TEXTDET_DET

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths to model checkpoints
    phase1_path = Path('runs/segmentation_v3/best.pt')
    phase2_path = Path('runs/detection_v5/best.pt')

    print(f"Loading Phase 1 from: {phase1_path}")
    print(f"Loading Phase 2 from: {phase2_path}")

    # ========== Phase 1 Model (for segmentation mask) ==========
    model_phase1 = create_text_detector(
        backbone_name='yolo11s.pt',
        pretrained_backbone=False,
        freeze_backbone=True,
        device=str(device)
    )

    # Load Phase 1 weights (backbone + seg_net)
    phase1_ckpt = torch.load(phase1_path, map_location=device, weights_only=False)
    # Clean checkpoint keys (remove _orig_mod. prefix from torch.compile)
    phase1_state = {k.replace('_orig_mod.', ''): v for k, v in phase1_ckpt['model_state_dict'].items()}
    model_phase1.load_state_dict(phase1_state, strict=False)
    model_phase1.eval()
    model_phase1.forward_mode = TEXTDET_MASK
    print("Phase 1 model loaded (segmentation only)")

    # ========== Phase 2 Model (for detection maps) ==========
    model_phase2 = create_text_detector(
        backbone_name='yolo11s.pt',
        pretrained_backbone=False,
        freeze_backbone=True,
        device=str(device)
    )

    # Load Phase 1 weights first (for backbone and seg_net)
    model_phase2.load_state_dict(phase1_state, strict=False)

    # Initialize DB head (this removes some seg_net layers for efficiency)
    model_phase2.initialize_db()

    # Load Phase 2 weights (DBNet)
    phase2_ckpt = torch.load(phase2_path, map_location=device, weights_only=False)
    state_dict = phase2_ckpt['model_state_dict']
    db_state = {}
    for k, v in state_dict.items():
        k = k.replace('_orig_mod.', '')
        if k.startswith('dbnet.'):
            db_state[k.replace('dbnet.', '')] = v

    if db_state:
        model_phase2.dbnet.load_state_dict(db_state)
        print("Phase 2 model loaded (detection)")
    else:
        # Try loading the full state dict if no separate dbnet keys
        model_phase2.load_state_dict(phase2_ckpt['model_state_dict'], strict=False)
        print("Phase 2 model loaded (full state dict)")

    # Move entire model to device (including dbnet created by initialize_db)
    model_phase2 = model_phase2.to(device)
    model_phase2.eval()
    model_phase2.forward_mode = TEXTDET_DET

    # ========== Find a test image ==========
    test_images = list(Path('data/val/images').glob('*.jpg'))[:1]
    if not test_images:
        test_images = list(Path('data/train/images').glob('*.jpg'))[:1]

    if not test_images:
        print("No test images found!")
        return

    test_img_path = test_images[0]
    print(f"Test image: {test_img_path}")

    # Load and preprocess image
    img = cv2.imread(str(test_img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    input_size = 1024
    img_resized = cv2.resize(img_rgb, (input_size, input_size))

    # Normalize
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # ========== Run inference ==========
    with torch.no_grad():
        # Phase 1: Segmentation
        seg_output = model_phase1(img_tensor)
        seg_mask = seg_output.squeeze().cpu().numpy()

        # Phase 2: Detection (DBNet)
        # In TEXTDET_DET mode, forward() already runs through dbnet and returns its output
        det_output = model_phase2(img_tensor)

        # det_output from DBHead is either:
        # - (prob_map, thresh_map, binary_map) tuple during training
        # - or a single tensor with 2 channels [prob_map, thresh_map] during inference
        print(f"det_output type: {type(det_output)}")
        if isinstance(det_output, tuple):
            print(f"det_output length: {len(det_output)}")
            for i, o in enumerate(det_output):
                print(f"  det_output[{i}] shape: {o.shape}")
            prob_map = det_output[0].squeeze().cpu().numpy()
            thresh_map = det_output[1].squeeze().cpu().numpy() if len(det_output) > 1 else prob_map
        else:
            print(f"det_output shape: {det_output.shape}")
            det_np = det_output.squeeze().cpu().numpy()
            # If shape is (2, H, W), first channel is prob_map, second is thresh_map
            if det_np.ndim == 3 and det_np.shape[0] == 2:
                prob_map = det_np[0]
                thresh_map = det_np[1]
            else:
                prob_map = det_np
                thresh_map = prob_map

    print(f"seg_mask shape: {seg_mask.shape}")
    print(f"prob_map shape: {prob_map.shape}")
    print(f"thresh_map shape: {thresh_map.shape}")
    print(f"Segmentation mask range: {seg_mask.min():.3f} - {seg_mask.max():.3f}")
    print(f"Detection prob_map range: {prob_map.min():.3f} - {prob_map.max():.3f}")
    print(f"Detection thresh_map range: {thresh_map.min():.3f} - {thresh_map.max():.3f}")

    # ========== Create visualization ==========
    fig_h, fig_w = 600, 2400
    canvas = np.ones((fig_h, fig_w, 3), dtype=np.uint8) * 255

    # Resize outputs for display
    display_size = 550
    img_display = cv2.resize(img_rgb, (display_size, display_size))

    # Segmentation mask visualization
    seg_mask_vis = (seg_mask * 255).astype(np.uint8)
    seg_mask_vis = cv2.resize(seg_mask_vis, (display_size, display_size))
    seg_mask_color = cv2.applyColorMap(seg_mask_vis, cv2.COLORMAP_JET)

    # Prob map visualization
    prob_vis = (prob_map * 255).astype(np.uint8)
    prob_vis = cv2.resize(prob_vis, (display_size, display_size))
    prob_color = cv2.applyColorMap(prob_vis, cv2.COLORMAP_JET)

    # Thresh map visualization
    thresh_vis = (thresh_map * 255).astype(np.uint8)
    thresh_vis = cv2.resize(thresh_vis, (display_size, display_size))
    thresh_color = cv2.applyColorMap(thresh_vis, cv2.COLORMAP_JET)

    # Place images on canvas
    y_offset = 25
    x_positions = [25, 600, 1175, 1750]

    canvas[y_offset:y_offset+display_size, x_positions[0]:x_positions[0]+display_size] = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
    canvas[y_offset:y_offset+display_size, x_positions[1]:x_positions[1]+display_size] = seg_mask_color
    canvas[y_offset:y_offset+display_size, x_positions[2]:x_positions[2]+display_size] = prob_color
    canvas[y_offset:y_offset+display_size, x_positions[3]:x_positions[3]+display_size] = thresh_color

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, 'Original', (x_positions[0] + 200, fig_h - 10), font, 0.8, (0, 0, 0), 2)
    cv2.putText(canvas, 'Phase1: Seg Mask', (x_positions[1] + 150, fig_h - 10), font, 0.8, (0, 0, 0), 2)
    cv2.putText(canvas, 'Phase2: Prob Map', (x_positions[2] + 150, fig_h - 10), font, 0.8, (0, 0, 0), 2)
    cv2.putText(canvas, 'Phase2: Thresh Map', (x_positions[3] + 130, fig_h - 10), font, 0.8, (0, 0, 0), 2)

    # Save
    output_dir = Path('results/images')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'phase1_phase2_test.jpg'
    cv2.imwrite(str(output_path), canvas)
    print(f"\nVisualization saved to: {output_path}")

if __name__ == '__main__':
    main()
