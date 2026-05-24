import zmq
from PIL import Image
import io
import numpy as np
import cv2
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

#Get two segmentations of the same scene. Check each mask against each other mask. Remove masks that overlap too much from the all-mask.

def track_two(image_1, image_2):
    pass

#Send it in both directions, keeping T-W and W-T. Then compare overlaps between the two T images and two W images.
#The ids that overlap above a threshold will have their masks put together as a single object for both T and W. 
#Mask by bits

def normalize_gsam_target_name(raw: str) -> str:
    t = (raw or "").strip().lower()
    if not t.endswith("."):
        t = t + "."
    return t


def send_segment_v2(
    socket: zmq.Socket,
    image: Image.Image,
    target_text: str,
    mode: str,
    params: Optional[Dict[str, Any]] = None,
    allow_fallback: bool = True,
    fallback_steps: Optional[list] = None,
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """
    New ZMQ protocol (pairs with server_gsam.py on port 8091):
      [b"segment", json_bytes, jpeg_bytes]
    Response: [json_bytes, mask_bytes] with mask uint8 (H,W) when ok.
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


def push_server_config(socket: zmq.Socket, params: Dict[str, Any]) -> Dict[str, Any]:
    socket.send_multipart([b"set_config", json.dumps(params).encode("utf-8")])
    return json.loads(socket.recv_multipart()[0].decode("utf-8"))


def remove_target_mask(instance_masks, target_mask, threshold=0.9):
    """
    Removes any mask in instance_masks that overlaps with the target_mask 
    by >= threshold (default 90%).

    Parameters:
    - instance_masks: np.ndarray of shape [num_masks, height, width]
                      Binary masks for each instance.
    - target_mask: np.ndarray of shape [height, width]
                   Binary mask to compare against.
    - threshold: float, overlap ratio to trigger removal.

    Returns:
    - np.ndarray: Filtered instance_masks with overlapping masks removed.
    """
    keep_masks = []
    target_area = np.sum(target_mask)

    for mask in instance_masks:
        intersection = np.sum(mask * target_mask)
        overlap_ratio = intersection / target_area if target_area > 0 else 0

        if overlap_ratio < threshold:
            keep_masks.append(mask)

    return np.stack(keep_masks) if keep_masks else np.zeros((0, *target_mask.shape), dtype=instance_masks.dtype)

def send_one(image, send_string, socket, force_segment=False, boxes_use=False):
    """
    Helper that maps a boolean segmentation/tracking intent to the new protocol.
    Returns a mask batch with shape (N, H, W), where N is 0 or 1.
    """
    if boxes_use:
        raise NotImplementedError("box return path not supported on current server reply.")
    mode = "segment" if force_segment else "track"
    meta, mask = send_segment_v2(socket, image, send_string, mode=mode)
    print(f"Received metadata: {meta}")
    if not meta.get("ok") or mask is None:
        return np.zeros((0, 1, 1), dtype=np.uint8)
    m = (mask > 127).astype(np.uint8) if mask.dtype == np.uint8 else (mask > 0.5).astype(np.uint8)
    if m.ndim == 2:
        m = m[np.newaxis, ...]
    return m

def instance_and_target_masks_to_one_mask(instance_mask, target_mask):
    """
    instance_mask = [masks, height, width] bool values
    target_mask = [height, width] bool values

    For each masks value, put it as the corresponding bit of a [height, width] as 1
    All values at mask0 move 0+1
    All values at mask1 move 1+1
    etc
    Keep the first as the target value

    return encoded_instance_mask = [masks] of 32bit values
    """
    encoded_instance_mask = np.zeros((instance_mask.shape[1], instance_mask.shape[2]), dtype=np.int32) #[height, width]
    for i in range(instance_mask.shape[0]):
        encoded_instance_mask |= (instance_mask[i].astype(np.int32) << (i+1))
    encoded_instance_mask |= (target_mask.astype(np.int32))

    return encoded_instance_mask

def send_instance_and_target(img, tar_string, socket):
    mask_instance = send_one(img, "Object.", socket, force_segment=True)
    mask_target = send_one(img, tar_string, socket, force_segment=True)
    if mask_target.shape[0] == 0:
        target_hw = np.zeros((1, 1), dtype=np.uint8)
    else:
        target_hw = mask_target[0]
    mask_instance = remove_target_mask(mask_instance, target_hw)
    if mask_instance.shape[0] == 0:
        encoded_instance_mask = target_hw.astype(np.int32)
    else:
        encoded_instance_mask = instance_and_target_masks_to_one_mask(
            mask_instance, target_hw
        )
    return mask_instance, mask_target, encoded_instance_mask


def collect_input_images() -> list[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    dataset_dir = Path("client_demo/dataset")
    
    roots = [dataset_dir]
    if not dataset_dir.exists():
        return []

    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for ext in exts:
            paths.extend(root.glob(ext))
        if paths:
            break

    return [str(p) for p in sorted(paths)]

def prep_and_send():
    context = zmq.Context()
    socket = context.socket(zmq.REQ) #Make sure to use REQ
    print("trying to connect to port")
    socket.connect("tcp://127.0.0.1:8091")
    print("Client connected")
    paths = collect_input_images()
    if not paths:
        print("No images found in the dataset directory.")
        return

    send_string = "Object."
    for i, image_path in enumerate(paths):
        image = Image.open(image_path)
        force_segment = False
        if i % 2 == 0: #Purposely want it to run object detection again when it is the first run, so I can keep the gsam running and continue restarting the object detection from there
            force_segment = True
        mode = "segment" if force_segment else "track"
        meta, mask = send_segment_v2(socket, image, send_string, mode=mode)
        print(meta)
        if mask is None:
            print("no mask")
            continue
        make_seg_img(np.stack([mask], axis=0), image_path, tag=i)

def make_seg_img(masks, image_path, tag=0, boxes=None):
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
    
    cv2.imwrite(f"./client_demo/output/segmented_{tag}.jpg", blended)

def main():
    prep_and_send()

if __name__ == "__main__":
    main()