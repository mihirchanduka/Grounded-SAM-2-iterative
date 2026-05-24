import zmq
from PIL import Image
import io
import numpy as np
import cv2
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

def normalize_gsam_target_name(raw: str) -> str:
    """Normalize a target phrase for Grounding DINO.

    Args:
        raw (str): The user-provided text prompt.

    Returns:
        str: The normalized prompt, lowercased and ending with a period.
    """
    t = (raw or "").strip().lower()
    if not t.endswith("."):
        t = t + "."
    return t

def send_segment_req(
    socket: zmq.Socket,
    image: Image.Image,
    target_text: str,
    mode: str,
    params: Optional[Dict[str, Any]] = None,
    allow_fallback: bool = True,
    fallback_steps: Optional[list[dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """
    New ZMQ protocol (pairs with server_gsam.py on port 8091):
      [b"segment", json_bytes, jpeg_bytes]
    Response: [json_bytes, mask_bytes] with mask uint8 (H,W) when ok.

    Args:
        socket (zmq.Socket): ZMQ REQ socket connected to the server.
        image (Image.Image): PIL image to send for segmentation.
        target_text (str): Text prompt for Grounding DINO.
        mode (str): Server mode, typically "segment" or "track".
        params (Optional[Dict[str, Any]]): Optional server-side segmentation parameters.
        allow_fallback (bool): Whether the server may relax thresholds.
        fallback_steps (Optional[list[dict[str, Any]]]): Optional fallback parameter sequence.

    Returns:
        Tuple[Dict[str, Any], Optional[np.ndarray]]: Response metadata and decoded mask, or None for the mask on failure.
    """
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    image_data = buf.getvalue()
    spec: Dict[str, Any] = {
        "target_text": normalize_gsam_target_name(target_text),
        "mode": mode,
        "params": params or {},
        "allow_fallback": allow_fallback,
    }
    if fallback_steps is not None:
        spec["fallback_steps"] = fallback_steps
    socket.send_multipart(
        [b"segment", json.dumps(spec).encode("utf-8"), image_data]
    )
    parts = socket.recv_multipart()
    meta = json.loads(parts[0].decode("utf-8"))
    if not meta.get("ok"):
        return meta, None
    mask = np.frombuffer(parts[1], dtype=np.dtype(meta["mask_dtype"])).reshape(
        tuple(meta["mask_shape"])
    )
    return meta, mask

def collect_input_image_groups(dataset_dir: Path) -> list[tuple[Path, list[Path]]]:
    """Collect dataset images grouped by their parent folder.

    Args:
        dataset_dir (Path): Root dataset directory.

    Returns:
        list[tuple[Path, list[Path]]]: Sorted folder groups with sorted image paths.
    """
    if not dataset_dir.exists():
        return []

    grouped_paths: dict[Path, list[Path]] = {}
    for image_path in sorted(dataset_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        rel_dir = image_path.parent.relative_to(dataset_dir)
        grouped_paths.setdefault(rel_dir, []).append(image_path)

    return [
        (folder, sorted(paths))
        for folder, paths in sorted(grouped_paths.items(), key=lambda item: str(item[0]))
    ]


def clean_output_dir(output_dir: Path) -> None:
    """Remove any previous demo outputs and recreate the directory.

    Args:
        output_dir (Path): Directory that stores rendered outputs.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

def segment_demo_images():
    """Run the demo client over the dataset images.
    """
    context = zmq.Context()
    socket = context.socket(zmq.REQ) #Make sure to use REQ
    print("Trying to connect to port")
    socket.connect("tcp://127.0.0.1:8091")
    print("Client connected")
    clean_output_dir(Path("client_demo/output"))
    grouped_paths = collect_input_image_groups(Path("client_demo/dataset"))
    if not grouped_paths:
        print("No images found in the dataset directory.")
        return

    send_string = "Red cheezit box."
    for folder_rel_path, image_paths in grouped_paths:
        print(f"\nProcessing folder: {folder_rel_path}")
        for index, image_path in enumerate(image_paths):
            image = Image.open(image_path)
            mode = "segment" if index == 0 else "track"
            meta, mask = send_segment_req(socket, image, send_string, mode=mode)
            print(meta)
            if mask is None:
                print("No mask")
                continue

            output_path = Path("client_demo/output") / folder_rel_path / f"segmented_{image_path.name}"
            make_seg_img(np.stack([mask], axis=0), image_path, output_path)

def make_seg_img(
    masks: np.ndarray,
    image_path: str,
    output_path: Path,
    boxes: Optional[np.ndarray] = None,
):
    """Render masks onto the source image and save the overlay.

    Args:
        masks (np.ndarray): Iterable of binary masks to draw.
        image_path (str): Path to the source image.
        boxes (Optional[np.ndarray]): Optional boxes to draw over the masks.
    """
    img = cv2.imread(image_path)
    overlay = np.zeros_like(img, dtype=np.uint8)
    i = -1
    for mask in masks:
        i += 1
        mask = (mask > 127).astype(np.uint8) if mask.dtype == np.uint8 else (mask > 0.5).astype(np.uint8)
        bold_colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Purple
            (0, 128, 255),    # Light Blue
            (128, 255, 0),    # Lime
            #(255, 0, 128),    # Pink
        ]
        color = np.array(random.choice(bold_colors), dtype=np.uint8)

        if i < len(bold_colors):
            color = np.array(bold_colors[i], dtype=np.uint8)
        else:
            color = tuple(np.random.randint(0, 256, size=3).tolist())
            color = np.array(color, dtype=np.uint8)
        #if i >= len(masks) - 1:
        #    print("Pink color")
        #    color = np.array((255, 0, 128), dtype=np.uint8) #Pink
        colored_mask = np.zeros_like(img, dtype=np.uint8)
        
        for c in range(3):
            colored_mask[:, :, c] = mask * color[c]
        
        overlay = np.where(mask[..., None], overlay + colored_mask, overlay)

        if boxes is not None:
            for box in boxes:
                x0, y0, x1, y1 = map(int, box)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), color=(0, 0, 0), thickness=2)

    # Clip values to avoid overflow after summation
    overlay = np.clip(overlay, 0, 255)

    # Blend once at the end
    blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), blended)

def main():
    """Entry point for the client demo.
    """
    segment_demo_images()

if __name__ == "__main__":
    main()