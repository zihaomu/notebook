"""Post-processing: overlay bounding boxes, labels, and VLM descriptions."""

import cv2
import numpy as np

import config

# Color palette for different classes
COLORS = [
    (255, 76, 76), (76, 255, 76), (76, 76, 255), (255, 255, 76),
    (255, 76, 255), (76, 255, 255), (255, 165, 76), (76, 165, 255),
    (165, 76, 255), (255, 76, 165), (76, 255, 165), (165, 255, 76),
]


def get_color(class_id):
    return COLORS[class_id % len(COLORS)]


def draw_detections(frame, detections, descriptions=None):
    """Draw bounding boxes, class labels, and VLM descriptions on frame.

    Args:
        frame: BGR numpy array (modified in-place)
        detections: list of (x1, y1, x2, y2, score, class_id)
        descriptions: optional dict mapping detection index to description string
    """
    desc_map = {}
    if descriptions:
        for det, desc in descriptions:
            key = (int(det[0]), int(det[1]), int(det[2]), int(det[3]))
            desc_map[key] = desc

    for det in detections:
        x1, y1, x2, y2, score, class_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = get_color(int(class_id))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        class_name = config.COCO_CLASSES[int(class_id)] if int(class_id) < len(config.COCO_CLASSES) else f"cls{int(class_id)}"
        label = f"{class_name} {score:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - baseline - 4), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        key = (x1, y1, x2, y2)
        if key in desc_map:
            _draw_vlm_text(frame, x1, y2, x2 - x1, desc_map[key], color)


def _draw_vlm_text(frame, x, y, box_width, text, color):
    """Draw VLM description text below the bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    max_width = max(box_width, 300)
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        (tw, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if tw > max_width and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)

    line_height = 22
    padding = 6
    total_height = len(lines) * line_height + 2 * padding
    max_text_width = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_text_width = max(max_text_width, tw)

    bg_x2 = min(x + max_text_width + 2 * padding, frame.shape[1])
    bg_y2 = min(y + total_height, frame.shape[0])

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    for i, line in enumerate(lines):
        ty = y + padding + (i + 1) * line_height - 4
        if ty < frame.shape[0]:
            cv2.putText(frame, line, (x + padding, ty), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(frame, line, (x + padding, ty), font, font_scale, (255, 255, 255), thickness)


def draw_scene_panel(frame, descriptions):
    """Draw a scene summary panel at the bottom of the frame.

    Shows all VLM descriptions in a single readable bar so the user can
    always see what the VLM is saying, even when per-box text is small.
    """
    if not descriptions:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    line_height = 24
    padding = 8
    margin = 4
    h, w = frame.shape[:2]

    lines = []
    for det, desc in descriptions:
        class_id = int(det[5])
        name = config.COCO_CLASSES[class_id] if class_id < len(config.COCO_CLASSES) else f"cls{class_id}"
        line = f"[{name}] {desc}"
        # truncate if too long for frame width
        while True:
            (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            if tw <= w - 2 * padding or len(line) <= 10:
                break
            line = line[:-4] + "..."
        lines.append(line)

    panel_h = len(lines) * line_height + 2 * padding
    panel_y = h - panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    for i, line in enumerate(lines):
        ty = panel_y + padding + (i + 1) * line_height - 6
        cv2.putText(frame, line, (padding, ty), font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, line, (padding, ty), font, font_scale, (200, 255, 200), thickness)


def draw_stats(frame, stats):
    """Draw performance stats overlay in top-left corner."""
    y = 30
    for key, value in stats.items():
        text = f"{key}: {value}"
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 25
