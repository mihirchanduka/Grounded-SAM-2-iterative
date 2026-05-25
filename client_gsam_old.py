import zmq
from PIL import Image
import io
import numpy as np
import cv2
import json
import shutil
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import supervision as sv

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
    target_text: Optional[str] = "None",
    mode: str = "segment",
    state_key: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    allow_fallback: bool = True,
    fallback_steps: Optional[list[dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """
    New ZMQ protocol (pairs with server_gsam.py on port 8091):
        [b"segment_best", json_bytes, jpeg_bytes]
    
    Response: [json_bytes, mask_bytes] with mask uint8 (H,W) when ok.

    Args:
        socket (zmq.Socket): ZMQ REQ socket connected to the server.
        image (Image.Image): PIL image to send for segmentation.
        target_text (Optional[str]): Text prompt for Grounding DINO.
        mode (str): Server mode, typically "segment" or "track".
        state_key (Optional[str]): State key used for tracking state lookup/storage.
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
        "mode": mode,
        "params": params or {},
        "allow_fallback": allow_fallback,
    }
    if target_text is not None:
        spec["target_text"] = normalize_gsam_target_name(target_text)
    if state_key is not None:
        spec["state_key"] = state_key
    if fallback_steps is not None:
        spec["fallback_steps"] = fallback_steps
    socket.send_multipart(
        [b"segment_best", json.dumps(spec).encode("utf-8"), image_data]
    )
    parts = socket.recv_multipart()
    meta = json.loads(parts[0].decode("utf-8"))
    if not meta.get("ok"):
        return meta, None
    mask = np.frombuffer(parts[1], dtype=np.dtype(meta["mask_dtype"])).reshape(
        tuple(meta["mask_shape"])
    )
    return meta, mask


def send_set_config_req(socket: zmq.Socket, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Send set_config and return response JSON.

    Args:
        socket (zmq.Socket): ZMQ REQ socket.
        updates (Dict[str, Any]): Partial config updates.

    Returns:
        Dict[str, Any]: Server reply JSON.
    """
    socket.send_multipart([b"set_config", json.dumps(updates).encode("utf-8")])
    parts = socket.recv_multipart()
    return json.loads(parts[0].decode("utf-8"))


def send_batch_segment_best_req(
    socket: zmq.Socket,
    image: Image.Image,
    requests: list[Dict[str, Any]],
) -> Tuple[Dict[str, Any], list[Optional[np.ndarray]]]:
    """Send batch_segment_best and decode all returned masks.

    Args:
        socket (zmq.Socket): ZMQ REQ socket.
        image (Image.Image): Input image.
        requests (list[Dict[str, Any]]): Per-request specs.

    Returns:
        Tuple[Dict[str, Any], list[Optional[np.ndarray]]]: Batch meta and masks aligned to results.
    """
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    image_data = buf.getvalue()

    socket.send_multipart(
        [
            b"batch_segment_best",
            json.dumps({"requests": requests}).encode("utf-8"),
            image_data,
        ]
    )
    parts = socket.recv_multipart()
    batch_meta = json.loads(parts[0].decode("utf-8"))
    results = batch_meta.get("results", [])

    masks: list[Optional[np.ndarray]] = []
    for result in results:
        idx = result.get("mask_part_index")
        if idx is None:
            masks.append(None)
            continue
        part_i = int(idx) + 1
        if part_i >= len(parts):
            masks.append(None)
            continue
        dtype = result.get("mask_dtype", "uint8")
        shape = tuple(result.get("mask_shape", []))
        if not shape:
            masks.append(None)
            continue
        mask = np.frombuffer(parts[part_i], dtype=np.dtype(dtype)).reshape(shape)
        masks.append(mask)

    return batch_meta, masks


def send_segment_instances_req(
    socket: zmq.Socket,
    image: Image.Image,
    target_text: str,
    params: Optional[Dict[str, Any]] = None,
    state_key: str = "",
) -> Tuple[Dict[str, Any], list[np.ndarray]]:
    """Send segment_instances and decode all instance masks.

    Args:
        socket (zmq.Socket): ZMQ REQ socket.
        image (Image.Image): Input image.
        target_text (str): Text prompt.
        params (Optional[Dict[str, Any]]): Optional params.
        state_key (str): Optional base state key.

    Returns:
        Tuple[Dict[str, Any], list[np.ndarray]]: Response JSON and decoded masks.
    """
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    image_data = buf.getvalue()

    spec = {
        "target_text": normalize_gsam_target_name(target_text),
        "params": params or {},
    }
    if state_key:
        spec["state_key"] = state_key
    socket.send_multipart([b"segment_instances", json.dumps(spec).encode("utf-8"), image_data])
    parts = socket.recv_multipart()
    meta = json.loads(parts[0].decode("utf-8"))
    if not meta.get("ok"):
        return meta, []

    masks: list[np.ndarray] = []
    for result in meta.get("results", []):
        idx = result.get("mask_part_index")
        if idx is None:
            continue
        part_i = int(idx) + 1
        if part_i >= len(parts):
            continue
        dtype = result.get("mask_dtype", "uint8")
        shape = tuple(result.get("mask_shape", []))
        if not shape:
            continue
        masks.append(np.frombuffer(parts[part_i], dtype=np.dtype(dtype)).reshape(shape))
    return meta, masks


def send_init_from_mask_req(
    socket: zmq.Socket,
    mask: np.ndarray,
    image: Image.Image,
    state_key: str,
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """Send init_from_mask and decode returned mask.

    Args:
        socket (zmq.Socket): ZMQ REQ socket.
        mask (np.ndarray): Input mask array.
        image (Image.Image): Input image.
        state_key (str): State key to write on server.

    Returns:
        Tuple[Dict[str, Any], Optional[np.ndarray]]: Response JSON and optional returned mask.
    """
    if image.mode == "RGBA":
        image = image.convert("RGB")
    if mask.dtype != np.uint8:
        mask_u8 = (mask > 0).astype(np.uint8) * 255
    else:
        mask_u8 = mask

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    image_data = buf.getvalue()

    spec = {
        "mask_dtype": "uint8",
        "mask_shape": list(mask_u8.shape),
        "state_key": state_key,
    }
    socket.send_multipart(
        [
            b"init_from_mask",
            json.dumps(spec).encode("utf-8"),
            mask_u8.tobytes(),
            image_data,
        ]
    )
    parts = socket.recv_multipart()
    meta = json.loads(parts[0].decode("utf-8"))
    if not meta.get("ok") or len(parts) < 2:
        return meta, None
    mask_shape = tuple(meta.get("mask_shape", []))
    mask_dtype = meta.get("mask_dtype", "uint8")
    if not mask_shape:
        return meta, None
    out_mask = np.frombuffer(parts[1], dtype=np.dtype(mask_dtype)).reshape(mask_shape)
    return meta, out_mask

def collect_input_images(dataset_dir: Path) -> list[Path]:
    """Collect dataset images from a single dataset directory.

    Args:
        dataset_dir (Path): Dataset directory.

    Returns:
        list[Path]: Sorted image paths.
    """
    if not dataset_dir.exists():
        return []

    image_paths: list[Path] = []
    for image_path in sorted(dataset_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            image_paths.append(image_path)
    return image_paths


def clean_output_dir(output_dir: Path) -> None:
    """Remove any previous demo outputs and recreate the directory.

    Args:
        output_dir (Path): Directory that stores rendered outputs.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _command_output_dir(output_root: Path, command_name: str) -> Path:
    """Return the output directory used for one command."""
    command_output_dir = output_root / command_name
    command_output_dir.mkdir(parents=True, exist_ok=True)
    return command_output_dir


def _frame_output_path(command_output_dir: Path, frame_path: Path) -> Path:
    """Return the output path for one frame under a command folder."""
    return command_output_dir / frame_path.name

def segment_demo_images(
    test_cmds: Optional[list[str]] = None,
    target_text: str = "Red cheezit box.",
    batch_target_text: Optional[list[str]] = None,
):
    """Run selected GSAM command tests over dataset images.

    Args:
        test_cmds (Optional[list[str]]): Command names to test.
        target_text (str): Prompt to use for text-driven commands.
        batch_target_text (Optional[list[str]]): Per-request target texts for batch_segment_best.
    """
    context = zmq.Context()
    socket = context.socket(zmq.REQ) #Make sure to use REQ
    print("Trying to connect to port")
    socket.connect("tcp://127.0.0.1:8091")
    print("Client connected")
    output_root = Path("client_demo/output")
    clean_output_dir(output_root)
    image_paths = collect_input_images(Path("client_demo/dataset"))
    if not image_paths:
        print("No images found in the dataset directory.")
        return

    all_cmds = [
        "set_config",
        "segment_best",
        "batch_segment_best",
        "segment_instances",
        "init_from_mask",
    ]
    selected = test_cmds or all_cmds
    invalid = [cmd for cmd in selected if cmd not in all_cmds]
    if invalid:
        raise ValueError(
            "Unknown command(s): " + ", ".join(invalid) + ". Valid values: " + ", ".join(all_cmds)
        )

    saved_mask_for_init: Optional[np.ndarray] = None

    if "set_config" in selected:
        print("\nTesting set_config")
        set_config_dir = _command_output_dir(output_root, "set_config")
        reply = send_set_config_req(socket, {"max_frames_in_state": 2})
        print(reply)
        with open(set_config_dir / "set_config_reply.json", "w", encoding="utf-8") as f:
            json.dump(reply, f, indent=2)

    if "segment_best" in selected:
        print("\nTesting segment_best on all frames")
        segment_best_dir = _command_output_dir(output_root, "segment_best")
        for frame_path in image_paths:
            with Image.open(frame_path) as image:
                meta, mask = send_segment_req(socket, image, target_text, mode="segment")
            print(frame_path.name, meta)
            if mask is None:
                continue
            if saved_mask_for_init is None:
                saved_mask_for_init = mask
            output_path = _frame_output_path(segment_best_dir, frame_path)
            make_seg_img(
                np.stack([mask], axis=0),
                str(frame_path),
                output_path,
                labels=[normalize_gsam_target_name(target_text).rstrip(".")],
            )

    if "batch_segment_best" in selected:
        print("\nTesting batch_segment_best on all frames")
        batch_texts = batch_target_text if batch_target_text else [target_text]
        reqs = [
            {
                "mode": "segment",
                "target_text": normalize_gsam_target_name(text),
                "params": {},
                "allow_fallback": True,
            }
            for text in batch_texts
        ]
        batch_segment_best_dir = _command_output_dir(output_root, "batch_segment_best")
        for frame_path in image_paths:
            with Image.open(frame_path) as image:
                meta, masks = send_batch_segment_best_req(socket, image, reqs)
            print(frame_path.name, meta)
            rendered_masks: list[np.ndarray] = []
            rendered_labels: list[str] = []
            for request, result, mask in zip(reqs, meta.get("results", []), masks):
                if mask is None:
                    continue
                rendered_masks.append(mask)
                rendered_labels.append(
                    result.get("label") or normalize_gsam_target_name(request["target_text"]).rstrip(".")
                )
            if rendered_masks:
                output_path = _frame_output_path(batch_segment_best_dir, frame_path)
                make_seg_img(
                    np.stack(rendered_masks, axis=0),
                    str(frame_path),
                    output_path,
                    labels=rendered_labels,
                )

    if "segment_instances" in selected:
        print("\nTesting segment_instances on all frames")
        segment_instances_dir = _command_output_dir(output_root, "segment_instances")
        for frame_path in image_paths:
            with Image.open(frame_path) as image:
                meta, masks = send_segment_instances_req(socket, image, target_text)
            print(frame_path.name, meta)
            if not masks:
                continue
            labels = [
                result.get("label")
                or f"{normalize_gsam_target_name(target_text).rstrip('.')} {result.get('object_id', i + 1)}"
                for i, result in enumerate(meta.get("results", []))
            ]
            output_path = _frame_output_path(segment_instances_dir, frame_path)
            make_seg_img(
                np.stack(masks, axis=0),
                str(frame_path),
                output_path,
                labels=labels,
            )

    if "init_from_mask" in selected:
        print("\nTesting init_from_mask")
        init_from_mask_dir = _command_output_dir(output_root, "init_from_mask")
        if saved_mask_for_init is None:
            # Bootstrap a mask from the first frame if segment_best did not run.
            with Image.open(image_paths[0]) as seed_image:
                seg_meta, seg_mask = send_segment_req(socket, seed_image, target_text, mode="segment")
            print("Bootstrap segment_best for init_from_mask:", seg_meta)
            saved_mask_for_init = seg_mask

        if saved_mask_for_init is None:
            print("Skipping init_from_mask: no seed mask available")
        else:
            skip_frames = 20
            seed_frame_path = image_paths[0]
            state_key = f"client_test_init_from_mask_{seed_frame_path.stem}"

            # Initialize tracking once from the first frame and seed mask.
            with Image.open(seed_frame_path) as seed_image:
                init_meta, init_mask = send_init_from_mask_req(
                    socket,
                    saved_mask_for_init,
                    seed_image,
                    state_key=state_key,
                )
            print(seed_frame_path.name, init_meta)
            if init_mask is not None:
                output_path = _frame_output_path(init_from_mask_dir, seed_frame_path)
                make_seg_img(
                    np.stack([init_mask], axis=0),
                    str(seed_frame_path),
                    output_path,
                    labels=["seeded instance"],
                )

            resume_start_idx = 1 + skip_frames
            if resume_start_idx >= len(image_paths):
                print(
                    f"Skipping resume tracking: only {len(image_paths)} frames available "
                    f"after skipping {skip_frames}"
                )
            else:
                print(f"Resuming tracking from frame index {resume_start_idx}")
                for frame_path in image_paths[resume_start_idx:]:
                    with Image.open(frame_path) as image:
                        track_meta, track_mask = send_segment_req(
                            socket,
                            image,
                            mode="track",
                            state_key=state_key,
                            target_text=None,
                        )
                    print(frame_path.name, track_meta)
                    if track_mask is None:
                        continue
                    output_path = _frame_output_path(init_from_mask_dir, frame_path)
                    make_seg_img(
                        np.stack([track_mask], axis=0),
                        str(frame_path),
                        output_path,
                        labels=["seeded instance"],
                    )

def make_seg_img(
    masks: np.ndarray,
    image_path: str,
    output_path: Path,
    boxes: Optional[np.ndarray] = None,
    labels: Optional[list[str]] = None,
):
    """Render masks onto the source image and save the overlay.

    Args:
        masks (np.ndarray): Iterable of binary masks to draw.
        image_path (str): Path to the source image.
        boxes (Optional[np.ndarray]): Optional boxes to draw over the masks.
        labels (Optional[list[str]]): Optional labels to render for each instance.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")

    mask_array = np.asarray(masks)
    if mask_array.ndim == 2:
        mask_array = mask_array[None, ...]
    mask_array = mask_array.astype(bool)
    if mask_array.size == 0:
        return

    if boxes is None:
        boxes = sv.mask_to_xyxy(mask_array)

    render_labels = list(labels) if labels is not None else []
    if len(render_labels) < len(mask_array):
        render_labels.extend(f"instance {i + 1}" for i in range(len(render_labels), len(mask_array)))
    render_labels = render_labels[: len(mask_array)]

    detections = sv.Detections(
        xyxy=boxes,
        mask=mask_array,
        class_id=np.arange(len(mask_array), dtype=np.int32),
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=render_labels,
    )

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated_frame)

def main():
    """Entry point for the client demo.
    """
    parser = argparse.ArgumentParser(description="GSAM client command tester")
    parser.add_argument(
        "--test-cmds",
        nargs="+",
        default=None,
        help=(
            "Commands to test. Valid: set_config segment_best batch_segment_best "
            "segment_instances init_from_mask"
        ),
    )
    parser.add_argument(
        "--target-text",
        default="red object.",
        help="Target text used by text-driven commands",
    )
    parser.add_argument(
        "--batch-target-text",
        nargs="+",
        default=["red cheezit box.", "blue can.", "red cylinder."],
        help="Target texts used by batch_segment_best, one request per string",
    )
    args = parser.parse_args()
    segment_demo_images(
        test_cmds=args.test_cmds,
        target_text=args.target_text,
        batch_target_text=args.batch_target_text,
    )

if __name__ == "__main__":
    main()