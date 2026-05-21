"""
GSAM server: ZMQ REP on port 8091.
Supported commands: ping | set_config | segment
"""

import io
import json
import traceback
from typing import Optional

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import load_single_image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

import zmq


# ---------------------------------------------------------------------------
# Box helpers
# ---------------------------------------------------------------------------

def _compute_area(box):
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


def _intersection_area(a, b):
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def _filter_boxes(boxes, overlap_threshold=0.9, max_inside=3):
    keep = []
    for i, A in enumerate(boxes):
        count = 0
        for j, B in enumerate(boxes):
            if i == j:
                continue
            area_B = _compute_area(B)
            if area_B > 0 and _intersection_area(A, B) / area_B >= overlap_threshold:
                count += 1
        if count <= max_inside:
            keep.append(A)
    return np.array(keep) if keep else np.zeros((0, 4), dtype=np.float32)


# ---------------------------------------------------------------------------
# Server params
# ---------------------------------------------------------------------------

def _default_server_params():
    return {
        "box_threshold": 0.20,
        "text_threshold": 0.25,
        "min_best_score": 0.35,
        "box_overlap_filter_threshold": 0.9,
        "max_inside": 3,
    }


def _merge_params(base: dict, override: Optional[dict]) -> dict:
    out = dict(base)
    if override:
        out.update(override)
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _match_label_to_target(label: str, target_text: str) -> bool:
    """True if a GDino detection label corresponds to a target text prompt."""
    lc = label.strip().rstrip(".").lower()
    tc = target_text.strip().rstrip(".").lower()
    if lc == tc or lc in tc:
        return True
    if tc in lc:
        return len(lc.split()) <= len(tc.split()) + 1
    return False


def segment_one(
    processor,
    grounding_model,
    video_predictor,
    device,
    text: str,
    image_pil: Image.Image,
    image_prepared,
    video_height: int,
    video_width: int,
    p: dict,
):
    """
    Run GDino + SAM2 for a single text prompt.
    Returns (mask_hw_bool, meta).  mask_hw_bool is None on failure.
    """
    meta = {
        "ok": False,
        "reason": "unknown",
        "best_score": None,
        "label": None,
        "params_used": dict(p),
    }

    # GDino detection
    inputs = processor(images=image_pil, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=float(p["box_threshold"]),
        text_threshold=float(p["text_threshold"]),
        target_sizes=[image_pil.size[::-1]],
    )

    scores = results[0]["scores"].cpu().numpy()
    boxes = results[0]["boxes"].cpu().numpy()
    labels = results[0]["labels"]
    print(f"  GDino '{text}': {len(boxes)} boxes  scores={np.round(scores, 3).tolist()}")

    if len(boxes) == 0:
        meta["reason"] = "no_boxes"
        return None, meta

    best_i = int(np.argmax(scores))
    best_score = float(scores[best_i])
    meta["best_score"] = best_score
    meta["label"] = str(labels[best_i]) if hasattr(labels, "__getitem__") else str(labels)

    if best_score < float(p["min_best_score"]):
        meta["reason"] = "low_score"
        return None, meta

    best_box = boxes[best_i : best_i + 1]
    filtered = _filter_boxes(
        best_box,
        overlap_threshold=float(p["box_overlap_filter_threshold"]),
        max_inside=int(p["max_inside"]),
    )
    if filtered.shape[0] == 0:
        meta["reason"] = "filtered_empty"
        return None, meta

    # SAM2 segmentation
    inference_state = video_predictor.non_video_path_init_state(
        image_prepared, video_height, video_width
    )
    video_predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        box=filtered[0],
    )
    video_segments = {}
    for frame_idx, obj_ids, mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[frame_idx] = {
            oid: (mask_logits[i] > 0.0).cpu().numpy()
            for i, oid in enumerate(obj_ids)
        }

    if not video_segments:
        meta["reason"] = "propagate_empty"
        return None, meta

    segments = next(iter(video_segments.values()))
    mask_list = list(segments.values())
    if not mask_list:
        meta["reason"] = "no_masks"
        return None, meta

    mask_hw = np.any(np.concatenate(mask_list, axis=0), axis=0).astype(np.bool_)
    meta.update({"ok": True, "reason": "ok"})
    return mask_hw, meta


def segment_with_fallback(
    processor,
    grounding_model,
    video_predictor,
    device,
    text: str,
    image_pil: Image.Image,
    image_prepared,
    video_height: int,
    video_width: int,
    base_params: dict,
    allow_fallback: bool,
    fallback_steps: list,
):
    """Attempt segmentation with base_params, then progressively looser thresholds."""
    attempts = [_merge_params(base_params, {})]
    if allow_fallback and fallback_steps:
        for step in fallback_steps:
            attempts.append(_merge_params(attempts[-1], step))

    last_meta = None
    for i, p in enumerate(attempts):
        mask_hw, meta = segment_one(
            processor, grounding_model, video_predictor, device,
            text, image_pil, image_prepared, video_height, video_width, p,
        )
        last_meta = meta
        if meta.get("ok") and mask_hw is not None:
            meta["used_fallback"] = i > 0
            if i > 0:
                print(f"  Fallback succeeded at attempt {i+1}/{len(attempts)}: {meta.get('params_used')}")
            return mask_hw, meta
        if i < len(attempts) - 1:
            print(f"  Fallback attempt {i+1} failed ({meta.get('reason')}); trying looser thresholds")

    last_meta = last_meta or {"ok": False, "reason": "unknown"}
    last_meta["ok"] = False
    last_meta["used_fallback"] = len(attempts) > 1
    print(f"  All attempts failed: reason={last_meta.get('reason')} best_score={last_meta.get('best_score')}")
    return None, last_meta


def batch_segment_fused(
    processor,
    grounding_model,
    video_predictor,
    device,
    batch_requests,  # list of (target_text, req_params, allow_fallback, fallback_steps)
    image_pil: Image.Image,
    image_prepared,
    video_height: int,
    video_width: int,
):
    """
    Single GDino forward pass for all N targets, then one shared SAM2 pass.
    Returns list of (mask_hw_bool|None, meta) in the same order as batch_requests.
    """
    target_texts = [r[0] for r in batch_requests]

    # "object." is a catch-all wildcard — exclude it from the GDino query so it doesn't
    # pollute token-span matching for the specific targets.  It will claim any unclaimed
    # detections after the specific targets are placed.
    _CATCHALL = "object"
    specific_texts = [t for t in target_texts if t.strip().rstrip(".").lower() != _CATCHALL]
    query_texts = specific_texts if specific_texts else target_texts
    combined_text = " ".join(t if t.endswith(".") else t + "." for t in query_texts)

    _GLOBAL_MIN_THR = 0.05
    inputs = processor(images=image_pil, text=combined_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=_GLOBAL_MIN_THR,
        text_threshold=_GLOBAL_MIN_THR,
        target_sizes=[image_pil.size[::-1]],
    )
    all_scores = results[0]["scores"].cpu().numpy()
    all_boxes = results[0]["boxes"].cpu().numpy()
    all_labels = [str(lbl) for lbl in results[0]["labels"]]
    print(f"[batch_fused] GDino: {len(all_labels)} detections for {len(query_texts)} specific targets")

    # Shared SAM2 image encoding
    inference_state = video_predictor.non_video_path_init_state(
        image_prepared, video_height, video_width
    )

    output_results = [None] * len(batch_requests)
    obj_id_map = {}  # obj_id -> (req_idx, best_score, label, p_used)
    next_obj_id = 1
    used_det_indices: set = set()

    for req_idx, (target_text, req_params, allow_fallback, fallback_steps) in enumerate(batch_requests):
        threshold_seq = [req_params]
        if allow_fallback and fallback_steps:
            threshold_seq.extend(fallback_steps)

        is_catchall = target_text.strip().rstrip(".").lower() == _CATCHALL
        if is_catchall:
            # Wildcard: match any detection not already claimed by a specific target.
            matching_dets = [
                (float(all_scores[i]), all_boxes[i], all_labels[i], i)
                for i in range(len(all_labels))
                if i not in used_det_indices
            ]
        else:
            matching_dets = [
                (float(all_scores[i]), all_boxes[i], all_labels[i], i)
                for i in range(len(all_labels))
                if _match_label_to_target(all_labels[i], target_text) and i not in used_det_indices
            ]

        if not matching_dets:
            output_results[req_idx] = (None, {"ok": False, "reason": "no_boxes", "best_score": None, "params_used": req_params})
            continue

        placed = False
        for level_idx, p in enumerate(threshold_seq):
            box_thr = float(p.get("box_threshold", 0.35))
            min_score = float(p.get("min_best_score", 0.35))
            candidates = [(s, b, l, idx) for s, b, l, idx in matching_dets if s >= box_thr]
            if not candidates:
                continue
            best_score, best_box, best_label, best_det_idx = max(candidates, key=lambda x: x[0])
            if best_score < min_score:
                continue
            used_det_indices.add(best_det_idx)
            obj_id = next_obj_id
            next_obj_id += 1
            video_predictor.add_new_points_or_box(
                inference_state=inference_state, frame_idx=0, obj_id=obj_id, box=best_box,
            )
            obj_id_map[obj_id] = (req_idx, best_score, best_label, p, level_idx > 0)
            placed = True
            break

        if not placed:
            best_overall = max((s for s, _, _, _ in matching_dets), default=None)
            output_results[req_idx] = (None, {"ok": False, "reason": "low_score", "best_score": best_overall, "params_used": threshold_seq[-1]})

    if not obj_id_map:
        for i in range(len(output_results)):
            if output_results[i] is None:
                output_results[i] = (None, {"ok": False, "reason": "unknown"})
        return output_results

    # One SAM2 propagation for all matched objects
    video_segments = {}
    for frame_idx, obj_ids, mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[frame_idx] = {
            oid: (mask_logits[i] > 0.0).cpu().numpy()
            for i, oid in enumerate(obj_ids)
        }
    segments = next(iter(video_segments.values())) if video_segments else {}

    for obj_id, (req_idx, best_score, label, p_used, used_fallback) in obj_id_map.items():
        mask_data = segments.get(obj_id)
        if mask_data is None:
            output_results[req_idx] = (None, {"ok": False, "reason": "propagate_no_mask", "best_score": best_score, "params_used": p_used})
            continue
        mask_hw = (np.any(mask_data, axis=0) if mask_data.ndim == 3 else mask_data).astype(np.bool_)
        output_results[req_idx] = (mask_hw, {"ok": True, "reason": "ok", "best_score": best_score, "label": label, "params_used": p_used, "used_fallback": used_fallback})

    for i in range(len(output_results)):
        if output_results[i] is None:
            output_results[i] = (None, {"ok": False, "reason": "unknown"})

    return output_results


# ---------------------------------------------------------------------------
# Reply helpers (always 2 parts: meta_json + mask_bytes)
# ---------------------------------------------------------------------------

def _make_meta_out(meta: dict, mask_hw: Optional[np.ndarray]) -> dict:
    ok = mask_hw is not None and bool(meta.get("ok"))
    return {
        "ok": ok,
        "reason": meta.get("reason", "ok" if ok else "error"),
        "used_fallback": bool(meta.get("used_fallback", False)),
        "params_used": meta.get("params_used"),
        "mask_dtype": "uint8" if ok else None,
        "mask_shape": list(mask_hw.shape) if ok else [],
        "best_score": meta.get("best_score"),
        "label": meta.get("label"),
    }


def _reply(socket, meta: dict, mask_hw: Optional[np.ndarray]):
    meta_out = _make_meta_out(meta, mask_hw)
    if mask_hw is not None and meta.get("ok"):
        socket.send_multipart([
            json.dumps(meta_out).encode("utf-8"),
            (mask_hw.astype(np.uint8) * 255).tobytes(),
        ])
    else:
        socket.send_multipart([json.dumps(meta_out).encode("utf-8"), b""])


def _reply_error(socket, reason: str):
    _reply(socket, {"ok": False, "reason": reason}, None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_FALLBACK_STEPS = [
    {"box_threshold": 0.35, "text_threshold": 0.25, "min_best_score": 0.35},
    {"box_threshold": 0.20, "text_threshold": 0.20, "min_best_score": 0.22},
    {"box_threshold": 0.12, "text_threshold": 0.18, "min_best_score": 0.16},
    {"box_threshold": 0.08, "text_threshold": 0.12, "min_best_score": 0.10},
]


_SAM2_MODELS = {
    "tiny":  ("./checkpoints/sam2.1_hiera_tiny.pt",  "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small": ("./checkpoints/sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base":  ("./checkpoints/sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("./checkpoints/sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Grounded-SAM-2 ZMQ server")
    parser.add_argument(
        "--model-size",
        choices=list(_SAM2_MODELS.keys()),
        default="small",
        help="SAM2 model size. 'tiny' is fastest; 'large' is most accurate.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8091,
        help="ZMQ REP port to bind.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Apply torch.compile to the grounding model for faster inference after warmup.",
    )
    args, _ = parser.parse_known_args()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://0.0.0.0:{args.port}")

    _is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        backend = "ROCm/HIP" if _is_rocm else "CUDA"
        print(f"Using device: {device} ({backend}) — {torch.cuda.get_device_name(0)}")
    else:
        print("Using device: cpu")

    if torch.cuda.is_available():
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.is_available() and not _is_rocm:
        props = torch.cuda.get_device_properties(0)
        if props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    sam2_checkpoint, model_cfg = _SAM2_MODELS[args.model_size]
    print(f"Loading SAM2 model: {args.model_size} ({sam2_checkpoint})")

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)  # noqa: F841 (kept for future use)

    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    if args.compile:
        print("Applying torch.compile to grounding model (first inference will be slow)...")
        grounding_model = torch.compile(grounding_model)

    server_defaults = _default_server_params()

    print(f"GSAM server ready on tcp://0.0.0.0:{args.port} (REP)")

    while True:
        message_parts = socket.recv_multipart()
        if not message_parts:
            continue

        cmd = message_parts[0]

        # ---- ping ----
        if cmd == b"ping":
            socket.send_multipart([
                json.dumps({
                    "ok": True,
                    "server": "grounded_sam_2",
                    "protocols": ["segment", "batch_segment", "set_config"],
                }).encode("utf-8")
            ])
            continue

        # ---- set_config ----
        if cmd == b"set_config":
            if len(message_parts) < 2:
                _reply_error(socket, "bad_set_config_request")
                continue
            try:
                upd = json.loads(message_parts[1].decode("utf-8"))
                server_defaults = _merge_params(server_defaults, upd)
                socket.send_multipart([
                    json.dumps({"ok": True, "applied": server_defaults}).encode("utf-8")
                ])
            except Exception as e:
                _reply_error(socket, f"set_config_error:{e}")
            continue

        # ---- segment ----
        if cmd == b"segment":
            if len(message_parts) < 3:
                _reply_error(socket, "bad_segment_request")
                continue
            try:
                spec = json.loads(message_parts[1].decode("utf-8"))
            except Exception as e:
                _reply_error(socket, f"bad_json:{e}")
                continue
            try:
                target_text = (spec.get("target_text") or "").strip().lower()
                if not target_text:
                    _reply_error(socket, "no_target_text")
                    continue
                if not target_text.endswith("."):
                    target_text += "."

                image_pil = Image.open(io.BytesIO(message_parts[2]))
                if image_pil.mode == "RGBA":
                    image_pil = image_pil.convert("RGB")
                image_prepared, video_height, video_width = load_single_image(
                    image_pil, 1024, compute_device=device
                )

                req_params = _merge_params(server_defaults, spec.get("params") or {})
                allow_fallback = bool(spec.get("allow_fallback", True))
                fallback_steps = spec.get("fallback_steps") or DEFAULT_FALLBACK_STEPS

                print(f"[segment] '{target_text}'")
                mask_hw, meta = segment_with_fallback(
                    processor, grounding_model, video_predictor, device,
                    target_text, image_pil, image_prepared,
                    video_height, video_width,
                    req_params, allow_fallback, fallback_steps,
                )
                _reply(socket, meta, mask_hw)

            except Exception as e:
                traceback.print_exc()
                _reply_error(socket, f"server_error:{e}")
            continue

        # ---- batch_segment ----
        if cmd == b"batch_segment":
            if len(message_parts) < 3:
                socket.send_multipart([json.dumps({"ok": False, "reason": "bad_batch_request", "results": []}).encode("utf-8")])
                continue
            try:
                spec = json.loads(message_parts[1].decode("utf-8"))
            except Exception as e:
                socket.send_multipart([json.dumps({"ok": False, "reason": f"bad_json:{e}", "results": []}).encode("utf-8")])
                continue
            try:
                requests_list = spec.get("requests") or []
                if not requests_list:
                    socket.send_multipart([json.dumps({"ok": False, "reason": "no_requests", "results": []}).encode("utf-8")])
                    continue

                image_pil = Image.open(io.BytesIO(message_parts[2]))
                if image_pil.mode == "RGBA":
                    image_pil = image_pil.convert("RGB")
                image_prepared, video_height, video_width = load_single_image(
                    image_pil, 1024, compute_device=device
                )

                batch_data = []
                for req in requests_list:
                    target_text = (req.get("target_text") or "").strip().lower()
                    if not target_text.endswith("."):
                        target_text += "."
                    req_params = _merge_params(server_defaults, req.get("params") or {})
                    allow_fallback = bool(req.get("allow_fallback", True))
                    fallback_steps = req.get("fallback_steps") or DEFAULT_FALLBACK_STEPS
                    batch_data.append((target_text, req_params, allow_fallback, fallback_steps))

                fused_out = batch_segment_fused(
                    processor, grounding_model, video_predictor, device,
                    batch_data, image_pil, image_prepared, video_height, video_width,
                )

                results = []
                mask_parts = []
                for mask_hw, meta in fused_out:
                    ok = mask_hw is not None and bool(meta.get("ok"))
                    meta_out = _make_meta_out(meta, mask_hw)
                    if ok:
                        meta_out["mask_part_index"] = len(mask_parts)
                        mask_parts.append((mask_hw.astype(np.uint8) * 255).tobytes())
                    results.append(meta_out)

                socket.send_multipart([
                    json.dumps({"ok": True, "results": results}).encode("utf-8"),
                    *mask_parts,
                ])
            except Exception as e:
                traceback.print_exc()
                socket.send_multipart([json.dumps({"ok": False, "reason": f"server_error:{e}", "results": []}).encode("utf-8")])
            continue

        # ---- unknown ----
        _reply_error(socket, f"unknown_command:{cmd.decode('utf-8', errors='replace')}")


if __name__ == "__main__":
    main()
