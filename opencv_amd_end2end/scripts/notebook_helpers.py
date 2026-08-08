"""Small display helpers shared by the pipeline notebooks."""

from __future__ import annotations

from pathlib import Path


def video_info(path: Path) -> dict[str, float | int]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    result = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(fps),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    result["duration_seconds"] = result["frames"] / result["fps"]
    capture.release()
    return result


def frame_at(path: Path, seconds: float):
    import cv2

    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read {path} at {seconds:.3f}s")
    return image


def show_bgr(image, title: str, size: tuple[int, int] = (14, 8)) -> None:
    import cv2
    import matplotlib.pyplot as plt

    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_bgr_grid(images, titles, size: tuple[int, int] = (20, 6)) -> None:
    import cv2
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(images), figsize=size)
    if len(images) == 1:
        axes = [axes]
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    plt.show()
