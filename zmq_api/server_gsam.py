import io
import json
import time
import uuid
import copy
import zlib
from typing import Any

import numpy as np
import torch
from PIL import Image
import zmq

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.misc import load_video_frames_from_pil_images
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


try:
    profile  # type: ignore[name-defined]
except NameError:
    def profile(func):  # type: ignore[no-redef]
        return func


SAM2_MODELS: dict[str, tuple[str, str]] = {
    "tiny":      ("./checkpoints/sam2.1_hiera_tiny.pt",      "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small":     ("./checkpoints/sam2.1_hiera_small.pt",     "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base_plus": ("./checkpoints/sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large":     ("./checkpoints/sam2.1_hiera_large.pt",     "configs/sam2.1/sam2.1_hiera_l.yaml"),
}


class ServerGSAM:
    @profile
    def __init__(
        self,
        endpoint: str = "tcp://0.0.0.0:8091",
        sam2_checkpoint: str = "./checkpoints/sam2.1_hiera_small.pt",
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
        compile_backbone: bool = True,
    ) -> None:
        self.endpoint = endpoint
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(self.endpoint)

        self.is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}{' (ROCm/HIP)' if self.is_rocm else ''}")

        if torch.cuda.is_available():
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.is_available() and not self.is_rocm:
            props = torch.cuda.get_device_properties(0)
            if props.major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        print(f"Loading SAM2: {sam2_checkpoint}")
        self.video_predictor = build_sam2_video_predictor(
            model_cfg, sam2_checkpoint, device=self.device
        )
        # Bound the per-state streaming memory: keep only the most recent N frames
        # of raw pixels + encoded memory so a long episode doesn't make
        # state_append_images O(n) per call (O(n^2) total) and leak GPU/CPU memory.
        # SAM2's attention reach is the floor: object pointers go back
        # max_obj_ptrs_in_encoder=16 frames, mask-memory ~10 (stride=2). 16 is the
        # exact minimum; 20 leaves a small margin. Set 0 to disable (legacy).
        self.video_predictor.streaming_keep_recent = 20
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)

        # torch.compile the heavy Hiera image backbone. Every track call re-encodes a
        # fixed image_size x image_size, batch-1 frame, so the shapes are static
        # (dynamic=False) — compile warms up once (first forward is slow) then runs
        # fused kernels with no accuracy change. Mirrors SAM2's built-in
        # compile_image_encoder path; we apply it post-build so it composes with the
        # track_multi shared-encode. Guarded per model so an unsupported build (e.g.
        # some ROCm setups) just falls back to eager. Skipped on CPU/ROCm.
        self._compiled_backbone = False
        if compile_backbone and self.device == "cuda" and not self.is_rocm:
            for label, model in (
                ("video", self.video_predictor),
                ("image", self.image_predictor.model),
            ):
                try:
                    model.image_encoder.forward = torch.compile(
                        model.image_encoder.forward,
                        mode="max-autotune",
                        fullgraph=True,
                        dynamic=False,
                    )
                    self._compiled_backbone = True
                    print(
                        f"Compiled {label} SAM2 image encoder "
                        "(torch.compile; first forward will be slow)."
                    )
                except Exception as exc:  # pragma: no cover
                    print(
                        f"torch.compile({label} image encoder) failed, "
                        f"using eager: {exc}"
                    )

        model_id = "IDEA-Research/grounding-dino-base"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(self.device)
        self.inference_state_dict: dict[str, Any] = {}

    def _state_metadata(self, state: Any) -> dict[str, Any]:
        if not isinstance(state, dict):
            return {
                "video_height": None,
                "video_width": None,
                "num_frames": None,
            }

        return {
            "video_height": state.get("video_height"),
            "video_width": state.get("video_width"),
            "num_frames": state.get("num_frames"),
        }

    def _generate_state_key(self) -> str:
        return uuid.uuid4().hex

    @staticmethod
    def _to_float_list(values: Any) -> list[float]:
        if values is None:
            return []
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        try:
            array = np.asarray(values, dtype=np.float32).reshape(-1)
        except Exception:
            return []
        return [float(value) for value in array.tolist()]

    def _json_or_error(self, raw: bytes) -> tuple[dict[str, Any] | None, str | None]:
        try:
            obj = json.loads(raw.decode("utf-8"))
            if not isinstance(obj, dict):
                return None, "spec_must_be_json_object"
            return obj, None
        except Exception as exc:  # pragma: no cover
            return None, f"bad_json:{exc}"

    def _reply(self, meta: dict[str, Any], parts: list[bytes] | None = None) -> None:
        if not parts:
            self.socket.send_multipart([json.dumps(meta).encode("utf-8")])
            return
        self.socket.send_multipart([json.dumps(meta).encode("utf-8"), *parts])

    @profile
    def _cuda_sync(self) -> None:
        """Block until queued GPU work finishes so perf_counter timings are real.

        CUDA kernels launch asynchronously, so without this the time of a GPU op
        gets misattributed to whatever later call happens to sync (e.g. a .cpu()).
        No-op on CPU. Cheap relative to the model work being timed.
        """
        if self.device == "cuda":
            torch.cuda.synchronize()

    def _segment_and_track_chunk(
        self,
        *,
        target_text: str,
        pil_images: list[Image.Image],
        box_threshold: float,
        text_threshold: float,
        start_frame_idx: int,
    ) -> tuple[dict[str, Any], list[bytes], Any | None]:
        if len(pil_images) == 0:
            return {"ok": False, "reason": "no_images"}, [], None

        if start_frame_idx < 0 or start_frame_idx >= len(pil_images):
            return {"ok": False, "reason": "bad_start_frame_idx"}, [], None

        # init = load the detect frame + spin up a fresh SAM2 inference state.
        t_init0 = time.perf_counter()
        det_image = pil_images[start_frame_idx]
        det_frame_tensor, video_height, video_width = load_video_frames_from_pil_images(
            [det_image],
            image_size=self.video_predictor.image_size,
            offload_video_to_cpu=True,
            compute_device=self.device,
        )

        inference_state = self.video_predictor.init_state_from_tensor_images(
            images=det_frame_tensor,
            video_height=video_height,
            video_width=video_width,
            offload_video_to_cpu=True,
        )
        self._cuda_sync()
        init_ms = (time.perf_counter() - t_init0) * 1e3

        # detect = GroundingDINO open-vocab box detection (the heavy part of a
        # "segment" call; this is why detect frames cost far more than tracking).
        t_det0 = time.perf_counter()
        det_inputs = self.processor(
            images=det_image, text=target_text, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            det_outputs = self.grounding_model(**det_inputs)

        det_results = self.processor.post_process_grounded_object_detection(
            det_outputs,
            det_inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[det_image.size[::-1]],
        )
        self._cuda_sync()
        detect_ms = (time.perf_counter() - t_det0) * 1e3

        boxes = det_results[0]["boxes"]
        labels = det_results[0]["labels"]
        det_scores = self._to_float_list(det_results[0].get("scores"))
        if boxes.shape[0] == 0:
            print(
                f"[server segment] text='{target_text}' boxes=0 | "
                f"init={init_ms:.0f}ms detect(GDINO)={detect_ms:.0f}ms (no boxes)",
                flush=True,
            )
            return {
                "ok": False,
                "reason": "no_boxes",
                "num_frames": len(pil_images),
                "num_instances": 0,
            }, [], None

        # sam = SAM2 mask prediction for the detected boxes + seeding the state.
        t_sam0 = time.perf_counter()
        self.image_predictor.set_image(np.array(det_image.convert("RGB")))
        masks, _, mask_scores = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
        mask_scores = self._to_float_list(mask_scores)

        if masks.ndim == 2:
            masks = masks[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        object_labels: dict[int, str] = {}
        object_confidences: dict[int, dict[str, Any]] = {}
        for object_id, mask in enumerate(masks, start=1):
            grounding_score = (
                det_scores[object_id - 1] if object_id - 1 < len(det_scores) else None
            )
            sam_score = (
                mask_scores[object_id - 1] if object_id - 1 < len(mask_scores) else None
            )
            confidence = sam_score if sam_score is not None else grounding_score
            self.video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=object_id,
                mask=mask,
            )
            object_labels[object_id] = str(labels[object_id - 1])
            object_confidences[object_id] = {
                "confidence": confidence,
                "grounding_score": grounding_score,
                "sam_score": sam_score,
            }
        self._cuda_sync()
        sam_ms = (time.perf_counter() - t_sam0) * 1e3
        print(
            f"[server segment] text='{target_text}' boxes={int(boxes.shape[0])} "
            f"instances={len(object_labels)} | init={init_ms:.0f}ms "
            f"detect(GDINO)={detect_ms:.0f}ms sam={sam_ms:.0f}ms",
            flush=True,
        )

        frame_parts: list[dict[str, Any]] = []
        raw_parts: list[bytes] = []

        # Build mask for the segmentation frame first.
        seed_mask_img = np.zeros((video_height, video_width), dtype=np.uint16)
        for object_id, mask in enumerate(masks, start=1):
            seed_mask_img[np.asarray(mask, dtype=bool)] = int(object_id)
        raw_parts.append(seed_mask_img.tobytes())
        frame_parts.append({"frame_idx": int(start_frame_idx), "mask_part_index": 0})

        # Track only the remaining forward frames by reusing the track helper.
        remaining_images = pil_images[start_frame_idx + 1 :]
        if len(remaining_images) > 0:
            track_meta, track_raw_parts, updated_state = self._track_chunk_from_state(
                inference_state=inference_state,
                pil_images=remaining_images,
            )
            if not track_meta.get("ok") or updated_state is None:
                return track_meta, [], None

            for i, chunk_part in enumerate(track_raw_parts):
                part_index = len(raw_parts)
                raw_parts.append(chunk_part)
                # _track_chunk_from_state frame indices are relative to the seed state.
                frame_parts.append(
                    {
                        "frame_idx": int(start_frame_idx + 1 + i),
                        "mask_part_index": part_index,
                    }
                )
            inference_state = updated_state

        meta = {
            "ok": True,
            "reason": "ok",
            "num_frames": len(frame_parts),
            "num_instances": len(object_labels),
            "instance_labels": object_labels,
            "instance_confidences": object_confidences,
            "mask_dtype": "uint16",
            "mask_shape": [int(video_height), int(video_width)],
            "frame_parts": frame_parts,
            "server_segment_detail": {
                "num_boxes": int(boxes.shape[0]),
                "init_ms": round(init_ms, 2),
                "detect_ms": round(detect_ms, 2),
                "sam_ms": round(sam_ms, 2),
            },
        }
        return meta, raw_parts, inference_state

    @profile
    def _track_chunk_from_state(
        self,
        *,
        inference_state: Any,
        pil_images: list[Image.Image],
        state_key: str = "",
        frames_tensor: Any | None = None,
        video_height: int | None = None,
        video_width: int | None = None,
        shared_feature: tuple[Any, Any] | None = None,
    ) -> tuple[dict[str, Any], list[bytes], Any | None]:
        if len(pil_images) == 0:
            return {"ok": False, "reason": "no_images"}, [], None

        # load = decode/resize the new frame(s) onto the device. track_multi
        # precomputes this ONCE and shares it across states (the frame is identical
        # for every state), so we skip the per-state reload when it's provided.
        t_load0 = time.perf_counter()
        if frames_tensor is None:
            frames_tensor, video_height, video_width = (
                load_video_frames_from_pil_images(
                    pil_images,
                    image_size=self.video_predictor.image_size,
                    offload_video_to_cpu=True,
                    compute_device=self.device,
                )
            )
        self._cuda_sync()
        load_ms = (time.perf_counter() - t_load0) * 1e3

        # mem_frames = how many frames are already in this state's SAM2 memory
        # bank BEFORE this call. Propagation attends over this history, so if
        # propagate_ms climbs as mem_frames grows (and resets when the episode
        # resets), the slowdown is memory-bank growth, not instance count.
        prev_num_frames = int(inference_state["num_frames"])
        t_app0 = time.perf_counter()
        self.video_predictor.state_append_images(
            inference_state,
            frames_tensor,
            video_height=video_height,
            video_width=video_width,
        )
        self._cuda_sync()
        append_ms = (time.perf_counter() - t_app0) * 1e3

        frame_parts: list[dict[str, Any]] = []
        raw_parts: list[bytes] = []
        object_confidences: dict[int, dict[str, Any]] = {}
        num_new_frames = int(frames_tensor.shape[0])

        # Reuse the Hiera backbone features computed once for this frame (track_multi
        # shares them across states). The new frame's absolute index is
        # prev_num_frames; seed this state's single-entry feature cache under that key
        # so propagate_in_video's _get_image_feature hits it instead of re-running the
        # encoder. The encoder is object-count-independent and dominates a small-N
        # track call, so this halves it when target + instance states are tracked.
        if shared_feature is not None and num_new_frames == 1:
            inference_state["cached_features"] = {prev_num_frames: shared_feature}

        # propagate = the actual SAM2 tracking forward pass (one per object,
        # attending over mem_frames of history). The .cpu() inside the loop syncs.
        t_prop0 = time.perf_counter()
        iterator = self.video_predictor.propagate_in_video(
            inference_state,
            start_frame_idx=prev_num_frames,
            max_frame_num_to_track=num_new_frames,
        )

        tracked_any = False
        for out_frame_idx, out_obj_ids, out_mask_logits in iterator:
            tracked_any = True
            # out_mask_logits is [N, 1, H, W] on the compute device. Do every
            # per-object reduction in batched tensor ops and cross the GPU->CPU
            # boundary ONCE. The old code called .any()/.item() per object, and each
            # of those forces a torch.cuda.synchronize() — so an N-object frame paid
            # ~2N hard syncs that serialized the GPU. That per-object sync tax is the
            # bulk of the Python overhead this loop adds on top of the SAM2 forward,
            # and it scales with object count (the thing we're trying to make cheap).
            logits = out_mask_logits[:, 0]                       # [N, H, W]
            masks = logits > 0.0                                 # [N, H, W] bool
            probs = torch.sigmoid(logits)                        # [N, H, W]
            areas = masks.flatten(1).sum(dim=1)                  # [N]
            fg_sums = (probs * masks).flatten(1).sum(dim=1)      # [N]
            # Per-object mean foreground prob over its own mask; for an empty mask
            # fall back to the whole-image mean (matches the old per-object branch).
            confs = torch.where(
                areas > 0,
                fg_sums / areas.clamp(min=1),
                probs.flatten(1).mean(dim=1),
            )
            confs_cpu = confs.cpu().tolist()                     # the single sync

            # Build the label image on the SAME device as the masks (the old code
            # allocated it on CPU and indexed it with GPU masks, which only worked
            # when inference ran on CPU); copy to host once at the end.
            mask_img = torch.zeros(
                video_height, video_width, dtype=torch.int32, device=logits.device
            )
            for i, out_obj_id in enumerate(out_obj_ids):
                # Last-writer-wins on overlap, same as the original loop order.
                mask_img[masks[i]] = int(out_obj_id)
                conf = float(confs_cpu[i])
                object_confidences[int(out_obj_id)] = {
                    "confidence": conf,
                    "grounding_score": None,
                    "sam_score": conf,
                }

            part_index = len(raw_parts)
            raw_parts.append(mask_img.cpu().numpy().astype(np.uint16).tobytes())
            frame_parts.append({"frame_idx": int(out_frame_idx), "mask_part_index": part_index})

        self._cuda_sync()
        propagate_ms = (time.perf_counter() - t_prop0) * 1e3
        num_objects = len(inference_state.get("obj_ids", []))
        # stored = physically retained frames (bounded by streaming_keep_recent);
        # mem_frames = logical frames seen so far. With the window on, `stored`
        # plateaus while `mem_frames` keeps climbing, and append/propagate stop
        # scaling. If `stored` keeps growing too, the bound is off.
        try:
            stored = int(inference_state["images"].shape[0])
        except Exception:
            stored = -1
        print(
            f"[server track] state={state_key or '?'} mem_frames={prev_num_frames} "
            f"stored={stored} objects={num_objects} new_frames={num_new_frames} | "
            f"load={load_ms:.0f}ms append={append_ms:.0f}ms "
            f"propagate={propagate_ms:.0f}ms tracked={tracked_any}",
            flush=True,
        )

        if not tracked_any:
            return {"ok": False, "reason": "tracking_failed"}, [], None

        meta = {
            "ok": True,
            "reason": "ok",
            "num_frames": len(frame_parts),
            "num_instances": num_objects,
            "instance_labels": {},
            "instance_confidences": object_confidences,
            "mask_dtype": "uint16",
            "mask_shape": [int(video_height), int(video_width)],
            "frame_parts": frame_parts,
            "server_track_detail": {
                "mem_frames": prev_num_frames,
                "num_objects": num_objects,
                "new_frames": num_new_frames,
                "load_ms": round(load_ms, 2),
                "append_ms": round(append_ms, 2),
                "propagate_ms": round(propagate_ms, 2),
            },
        }
        return meta, raw_parts, inference_state

    @profile
    def _handle_segment_instances(self, message_parts: list[bytes]) -> None:
        t_handler_start = time.perf_counter()
        if len(message_parts) < 3:
            self._reply({"ok": False, "reason": "missing_spec_or_images"})
            return

        spec, parse_err = self._json_or_error(message_parts[1])
        if parse_err:
            self._reply({"ok": False, "reason": parse_err})
            return

        mode = str(spec.get("mode", "segment")).strip().lower() if spec else "segment"
        state_key = str(spec.get("state_key", "")).strip() if spec else ""
        if not state_key:
            state_key = self._generate_state_key()

        t_decode0 = time.perf_counter()
        try:
            pil_images = [
                Image.open(io.BytesIO(part)).convert("RGB")
                for part in message_parts[2:]
            ]
        except Exception as exc:
            self._reply({"ok": False, "reason": f"bad_image_bytes:{exc}"})
            return
        img_decode_ms = (time.perf_counter() - t_decode0) * 1e3
        # Only compress/shrink masks when the client advertised support, so an
        # older client (no flag) still gets the legacy uint16 raw payload.
        accept_compression = bool(spec.get("accept_compression")) if spec else False

        def _finish(meta: dict[str, Any], compute_ms: float, raw_parts) -> None:
            # Shrink the returned label masks (uint16 raw -> uint8 + zlib) before
            # the timing snapshot, since that compression is real server work and
            # the whole point is to cut wire bytes. A label image is mostly one
            # value, so zlib typically shrinks it 10-50x for ~1-3ms.
            if accept_compression and raw_parts:
                raw_parts = self._compress_mask_parts(meta, raw_parts)
            # Timing the server pays (excludes ZMQ wire time).
            meta["mode_timed"] = mode
            meta["server_img_decode_ms"] = round(img_decode_ms, 2)
            meta["server_compute_ms"] = round(compute_ms, 2)
            meta["server_total_ms"] = round(
                (time.perf_counter() - t_handler_start) * 1e3, 2
            )
            print(
                f"[server handler] cmd={mode} state={meta.get('state_key', '?')} "
                f"ok={meta.get('ok')} | img_decode={img_decode_ms:.0f}ms "
                f"compute={compute_ms:.0f}ms total={meta['server_total_ms']:.0f}ms",
                flush=True,
            )
            self._reply(meta, raw_parts)

        if mode == "segment":
            target_text = str(spec.get("target_text", "")).strip()
            if not target_text:
                self._reply({"ok": False, "reason": "missing_target_text"})
                return

            box_threshold = float(spec.get("box_threshold", 0.25))
            text_threshold = float(spec.get("text_threshold", 0.25))
            # Segment on the first image, then track the rest of the chunk.
            t_c0 = time.perf_counter()
            meta, raw_parts, inference_state = self._segment_and_track_chunk(
                target_text=target_text,
                pil_images=pil_images,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                start_frame_idx=0,
            )
            compute_ms = (time.perf_counter() - t_c0) * 1e3
            if meta.get("ok") and inference_state is not None:
                self.inference_state_dict[state_key] = inference_state
            meta["state_key"] = state_key
            _finish(meta, compute_ms, raw_parts)
        elif mode == "track":
            existing_state = self.inference_state_dict.get(state_key)
            if existing_state is None:
                self._reply({"ok": False, "reason": "state_not_found", "state_key": state_key})
                return
            t_c0 = time.perf_counter()
            meta, raw_parts, updated_state = self._track_chunk_from_state(
                inference_state=existing_state,
                pil_images=pil_images,
                state_key=state_key,
            )
            compute_ms = (time.perf_counter() - t_c0) * 1e3
            if meta.get("ok") and updated_state is not None:
                self.inference_state_dict[state_key] = updated_state
            meta["state_key"] = state_key
            _finish(meta, compute_ms, raw_parts)
        else:
            self._reply({"ok": False, "reason": "invalid_mode", "state_key": state_key})

    @staticmethod
    def _compress_mask_parts(meta: dict[str, Any], raw_parts: list[bytes]) -> list[bytes]:
        """uint16 raw mask bytes -> uint8 (when ids fit) + zlib. Updates meta in place.

        Centralized here so the segment/track builders stay unchanged; the cost is
        one extra frombuffer + a cheap dtype cast on the server, far less than the
        wire time it saves. Falls back to no-op (returns input) if anything is off.
        """
        shape = meta.get("mask_shape")
        if not shape or len(shape) != 2:
            return raw_parts
        src_dtype = np.dtype(meta.get("mask_dtype", "uint16"))
        try:
            arrs = [np.frombuffer(p, dtype=src_dtype).reshape(shape) for p in raw_parts]
        except Exception:
            return raw_parts
        max_id = max((int(a.max(initial=0)) for a in arrs), default=0)
        out_dtype = np.uint8 if max_id < 256 else np.uint16
        compressed = [
            zlib.compress(a.astype(out_dtype, copy=False).tobytes(), 1) for a in arrs
        ]
        meta["mask_dtype"] = "uint8" if out_dtype is np.uint8 else "uint16"
        meta["mask_compression"] = "zlib"
        return compressed

    @profile
    def _handle_track_multi(self, message_parts: list[bytes]) -> None:
        """Track several inference states on ONE uploaded image in a single
        round-trip. The client uses this to fetch a camera's target + instance
        masks together instead of two separate calls (one image upload, one RTT).
        Reply: meta.results[i] = {state_key, ok, mask_part_index, ...}; raw_parts
        holds the (optionally compressed) masks indexed by mask_part_index.
        """
        t_handler_start = time.perf_counter()
        if len(message_parts) < 3:
            self._reply({"ok": False, "reason": "missing_spec_or_images"})
            return
        spec, parse_err = self._json_or_error(message_parts[1])
        if parse_err:
            self._reply({"ok": False, "reason": parse_err})
            return
        state_keys = spec.get("state_keys", []) if spec else []
        if not isinstance(state_keys, list) or not state_keys:
            self._reply({"ok": False, "reason": "missing_state_keys"})
            return
        accept_compression = bool(spec.get("accept_compression")) if spec else False

        t_decode0 = time.perf_counter()
        try:
            pil_images = [
                Image.open(io.BytesIO(part)).convert("RGB")
                for part in message_parts[2:]
            ]
        except Exception as exc:
            self._reply({"ok": False, "reason": f"bad_image_bytes:{exc}"})
            return
        img_decode_ms = (time.perf_counter() - t_decode0) * 1e3

        # Load + encode the shared frame ONCE for all states. The image is identical
        # across the target/instance states, and the Hiera backbone is the dominant,
        # object-count-independent cost of a small-N track call — running it per state
        # doubled it. Each state still runs its own per-object memory attention +
        # mask decode; only the encoder is shared. Falls back to per-state load/encode
        # if anything here fails.
        frames_tensor = video_height = video_width = shared_feature = None
        t_enc0 = time.perf_counter()
        try:
            frames_tensor, video_height, video_width = (
                load_video_frames_from_pil_images(
                    pil_images,
                    image_size=self.video_predictor.image_size,
                    offload_video_to_cpu=True,
                    compute_device=self.device,
                )
            )
            if int(frames_tensor.shape[0]) == 1:
                new_image = frames_tensor[0].to(self.device).float().unsqueeze(0)
                with torch.inference_mode():
                    backbone_out = self.video_predictor.forward_image(new_image)
                shared_feature = (new_image, backbone_out)
            self._cuda_sync()
        except Exception as exc:  # pragma: no cover
            print(
                f"[track_multi] shared frame encode failed, per-state fallback: {exc}",
                flush=True,
            )
            frames_tensor = video_height = video_width = shared_feature = None
        shared_encode_ms = (time.perf_counter() - t_enc0) * 1e3

        results: list[dict[str, Any]] = []
        raw_parts: list[bytes] = []
        mask_shape = None
        compute_ms = 0.0
        for raw_key in state_keys:
            sk = str(raw_key).strip()
            existing = self.inference_state_dict.get(sk)
            if existing is None:
                results.append(
                    {"state_key": sk, "ok": False, "reason": "state_not_found",
                     "mask_part_index": None}
                )
                continue
            t_c0 = time.perf_counter()
            meta_i, raw_i, updated = self._track_chunk_from_state(
                inference_state=existing, pil_images=pil_images, state_key=sk,
                frames_tensor=frames_tensor, video_height=video_height,
                video_width=video_width, shared_feature=shared_feature,
            )
            compute_ms += (time.perf_counter() - t_c0) * 1e3
            if not meta_i.get("ok") or updated is None or not raw_i:
                results.append(
                    {"state_key": sk, "ok": False,
                     "reason": meta_i.get("reason", "track_failed"),
                     "mask_part_index": None}
                )
                continue
            self.inference_state_dict[sk] = updated
            mask_shape = meta_i.get("mask_shape")
            part_index = len(raw_parts)
            raw_parts.append(raw_i[0])  # single image -> single mask part per state
            results.append(
                {"state_key": sk, "ok": True, "reason": "ok",
                 "num_instances": meta_i.get("num_instances", 0),
                 "instance_labels": meta_i.get("instance_labels", {}),
                 "instance_confidences": meta_i.get("instance_confidences", {}),
                 "mask_part_index": part_index}
            )

        meta: dict[str, Any] = {
            "ok": any(r["ok"] for r in results),
            "results": results,
            "mask_dtype": "uint16",
            "mask_shape": mask_shape,
        }
        if accept_compression and raw_parts:
            raw_parts = self._compress_mask_parts(meta, raw_parts)
        meta["mode_timed"] = "track_multi"
        meta["server_img_decode_ms"] = round(img_decode_ms, 2)
        meta["server_shared_encode_ms"] = round(shared_encode_ms, 2)
        meta["server_compute_ms"] = round(compute_ms, 2)
        meta["server_total_ms"] = round(
            (time.perf_counter() - t_handler_start) * 1e3, 2
        )
        ok_states = sum(1 for r in results if r.get("ok"))
        print(
            f"[server handler] cmd=track_multi states={len(state_keys)} "
            f"ok_states={ok_states} | img_decode={img_decode_ms:.0f}ms "
            f"shared_encode={shared_encode_ms:.0f}ms "
            f"compute={compute_ms:.0f}ms total={meta['server_total_ms']:.0f}ms",
            flush=True,
        )
        self._reply(meta, raw_parts)

    def _handle_copy_state(self, message_parts: list[bytes]) -> None:
        if len(message_parts) < 2:
            self._reply({"ok": False, "reason": "missing_spec"})
            return

        spec, parse_err = self._json_or_error(message_parts[1])
        if parse_err:
            self._reply({"ok": False, "reason": parse_err})
            return

        source_state_key = str(spec.get("source_state_key", "")).strip() if spec else ""
        if not source_state_key:
            self._reply({"ok": False, "reason": "missing_source_state_key"})
            return

        source_state = self.inference_state_dict.get(source_state_key)
        if source_state is None:
            self._reply(
                {
                    "ok": False,
                    "reason": "source_state_not_found",
                    "source_state_key": source_state_key,
                }
            )
            return

        new_state_key = str(spec.get("new_state_key", "")).strip() if spec else ""
        if not new_state_key:
            new_state_key = self._generate_state_key()

        if new_state_key in self.inference_state_dict:
            self._reply(
                {
                    "ok": False,
                    "reason": "target_state_key_exists",
                    "source_state_key": source_state_key,
                    "state_key": new_state_key,
                }
            )
            return

        self.inference_state_dict[new_state_key] = copy.deepcopy(source_state)
        self._reply(
            {
                "ok": True,
                "reason": "ok",
                "source_state_key": source_state_key,
                "state_key": new_state_key,
            }
        )

    def _handle_remove_state(self, message_parts: list[bytes]) -> None:
        if len(message_parts) < 2:
            self._reply({"ok": False, "reason": "missing_spec"})
            return

        spec, parse_err = self._json_or_error(message_parts[1])
        if parse_err:
            self._reply({"ok": False, "reason": parse_err})
            return

        state_key = str(spec.get("state_key", "")).strip() if spec else ""
        if not state_key:
            self._reply({"ok": False, "reason": "missing_state_key"})
            return

        if state_key not in self.inference_state_dict:
            self._reply({"ok": False, "reason": "state_not_found", "state_key": state_key})
            return

        del self.inference_state_dict[state_key]
        self._reply(
            {
                "ok": True,
                "reason": "ok",
                "state_key": state_key,
                "num_states": len(self.inference_state_dict),
            }
        )

    def _handle_prune_objects(self, message_parts: list[bytes]) -> None:
        """Remove specific object ids from a tracked state so future propagation
        stops spending time on them. The perception node calls this once its 3D
        filters decide a tracked instance is really the floor/wall/shelf or a coarse
        mask that merged two objects. Propagation cost is ~linear in object count, so
        dropping the junk keeps tracking proportional to the REAL object count."""
        if len(message_parts) < 2:
            self._reply({"ok": False, "reason": "missing_spec"})
            return

        spec, parse_err = self._json_or_error(message_parts[1])
        if parse_err:
            self._reply({"ok": False, "reason": parse_err})
            return

        state_key = str(spec.get("state_key", "")).strip() if spec else ""
        if not state_key:
            self._reply({"ok": False, "reason": "missing_state_key"})
            return

        state = self.inference_state_dict.get(state_key)
        if state is None:
            self._reply(
                {"ok": False, "reason": "state_not_found", "state_key": state_key}
            )
            return

        try:
            obj_ids = [int(o) for o in (spec.get("obj_ids") or [])]
        except (TypeError, ValueError):
            self._reply({"ok": False, "reason": "bad_obj_ids", "state_key": state_key})
            return
        if not obj_ids:
            self._reply({"ok": False, "reason": "no_obj_ids", "state_key": state_key})
            return

        removed: list[int] = []
        remaining: list[int] = list(state.get("obj_ids", []))
        for obj_id in obj_ids:
            try:
                # need_output=False: we don't need recomputed masks back, just the
                # object gone from the state so the next propagate skips it.
                remaining, _ = self.video_predictor.remove_object(
                    state, obj_id, strict=False, need_output=False
                )
                removed.append(obj_id)
            except Exception as exc:  # pragma: no cover
                print(
                    f"[server prune] remove_object({obj_id}) failed: {exc}", flush=True
                )
        print(
            f"[server prune] state={state_key} requested={obj_ids} "
            f"removed={removed} remaining={list(remaining)}",
            flush=True,
        )
        self._reply(
            {
                "ok": True,
                "reason": "ok",
                "state_key": state_key,
                "removed": removed,
                "remaining_obj_ids": [int(o) for o in remaining],
                "num_objects": len(remaining),
            }
        )

    def _handle_remove_all_states(self, message_parts: list[bytes]) -> None:
        # This command accepts no required payload and clears all cached states.
        _ = message_parts
        removed_count = len(self.inference_state_dict)
        self.inference_state_dict.clear()
        self._reply(
            {
                "ok": True,
                "reason": "ok",
                "removed_count": removed_count,
                "num_states": 0,
            }
        )

    def _handle_list_states(self, message_parts: list[bytes]) -> None:
        _ = message_parts
        states: dict[str, Any] = {}
        for state_key, state in self.inference_state_dict.items():
            states[str(state_key)] = self._state_metadata(state)

        self._reply(
            {
                "ok": True,
                "reason": "ok",
                "num_states": len(states),
                "states": states,
            }
        )

    @profile
    def run(self) -> None:
        print(
            f"GSAM server ready on {self.endpoint} (REP), "
            "commands=segment_instances,track_multi,copy_state,remove_state,"
            "prune_objects,remove_all_states,list_states"
        )
        while True:
            message_parts = self.socket.recv_multipart(flags=0)
            if not message_parts:
                self._reply({"ok": False, "reason": "empty_request"})
                continue

            cmd = message_parts[0]
            if cmd == b"segment_instances":
                self._handle_segment_instances(message_parts)
                continue

            if cmd == b"track_multi":
                self._handle_track_multi(message_parts)
                continue

            if cmd == b"copy_state":
                self._handle_copy_state(message_parts)
                continue

            if cmd == b"remove_state":
                self._handle_remove_state(message_parts)
                continue

            if cmd == b"prune_objects":
                self._handle_prune_objects(message_parts)
                continue

            if cmd == b"remove_all_states":
                self._handle_remove_all_states(message_parts)
                continue

            if cmd == b"list_states":
                self._handle_list_states(message_parts)
                continue

            self._reply({"ok": False, "reason": f"unknown_command:{cmd!r}"})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GSAM ZMQ server")
    parser.add_argument(
        "--model",
        choices=list(SAM2_MODELS.keys()),
        default="small",
        help="SAM2 model variant (default: small)",
    )
    parser.add_argument(
        "--endpoint",
        default="tcp://0.0.0.0:8091",
        help="ZMQ REP endpoint to bind (default: tcp://0.0.0.0:8091)",
    )
    parser.add_argument(
        "--no-compile",
        dest="compile_backbone",
        action="store_false",
        help=(
            "Disable torch.compile of the SAM2 image encoder. On by default (CUDA, "
            "non-ROCm); compile makes the first forward slow but speeds up every "
            "track afterward. Use this if compile is unstable on your setup."
        ),
    )
    args = parser.parse_args()

    checkpoint, cfg = SAM2_MODELS[args.model]
    server = ServerGSAM(
        endpoint=args.endpoint,
        sam2_checkpoint=checkpoint,
        model_cfg=cfg,
        compile_backbone=args.compile_backbone,
    )
    server.run()
