"""Minimal GSAM ZMQ server exposing only `segment_instances`.

Protocol
- Endpoint: tcp://0.0.0.0:8091 (REP)
- Request multipart:
  1) b"segment_instances"
  2) JSON spec bytes
  3..N) image bytes (each frame is any PIL-readable image)

Spec JSON fields (all optional unless noted)
- target_text (str, required): grounding text prompt, e.g. "car."
- box_threshold (float, default 0.25)
- text_threshold (float, default 0.25)
- start_frame_idx (int, default 0): index in provided chunk used for grounding

Reply multipart on success
  1) JSON metadata
  2..M) one uint16 instance mask per frame (shape HxW), indexed by `frame_parts`

Reply multipart on failure
  1) JSON metadata only
"""

import io
import json
import uuid
from typing import Any

import numpy as np
import torch
from PIL import Image
import zmq

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.misc import load_video_frames_from_pil_images
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


SAM2_MODELS: dict[str, tuple[str, str]] = {
    "tiny":      ("./checkpoints/sam2.1_hiera_tiny.pt",      "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small":     ("./checkpoints/sam2.1_hiera_small.pt",     "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base_plus": ("./checkpoints/sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large":     ("./checkpoints/sam2.1_hiera_large.pt",     "configs/sam2.1/sam2.1_hiera_l.yaml"),
}


class ServerGSAM:
    def __init__(
        self,
        endpoint: str = "tcp://0.0.0.0:8091",
        sam2_checkpoint: str = "./checkpoints/sam2.1_hiera_small.pt",
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
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
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)

        model_id = "IDEA-Research/grounding-dino-base"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(self.device)
        self.inference_state_dict: dict[str, Any] = {}

    def _generate_state_key(self) -> str:
        return uuid.uuid4().hex

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

        boxes = det_results[0]["boxes"]
        labels = det_results[0]["labels"]
        if boxes.shape[0] == 0:
            return {
                "ok": False,
                "reason": "no_boxes",
                "num_frames": len(pil_images),
                "num_instances": 0,
            }, [], None

        self.image_predictor.set_image(np.array(det_image.convert("RGB")))
        masks, _, _ = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )

        if masks.ndim == 2:
            masks = masks[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        object_labels: dict[int, str] = {}
        for object_id, mask in enumerate(masks, start=1):
            self.video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=object_id,
                mask=mask,
            )
            object_labels[object_id] = str(labels[object_id - 1])

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
            "mask_dtype": "uint16",
            "mask_shape": [int(video_height), int(video_width)],
            "frame_parts": frame_parts,
        }
        return meta, raw_parts, inference_state

    def _track_chunk_from_state(
        self,
        *,
        inference_state: Any,
        pil_images: list[Image.Image],
    ) -> tuple[dict[str, Any], list[bytes], Any | None]:
        if len(pil_images) == 0:
            return {"ok": False, "reason": "no_images"}, [], None

        frames_tensor, video_height, video_width = load_video_frames_from_pil_images(
            pil_images,
            image_size=self.video_predictor.image_size,
            offload_video_to_cpu=True,
            compute_device=self.device,
        )

        prev_num_frames = int(inference_state["num_frames"])
        self.video_predictor.state_append_images(
            inference_state,
            frames_tensor,
            video_height=video_height,
            video_width=video_width,
        )

        frame_parts: list[dict[str, Any]] = []
        raw_parts: list[bytes] = []
        num_new_frames = int(frames_tensor.shape[0])
        
        iterator = self.video_predictor.propagate_in_video(
            inference_state,
            start_frame_idx=prev_num_frames,
            max_frame_num_to_track=num_new_frames,
        )

        tracked_any = False
        for out_frame_idx, out_obj_ids, out_mask_logits in iterator:
            tracked_any = True
            mask_img = torch.zeros(video_height, video_width, dtype=torch.int32)
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = out_mask_logits[i] > 0.0
                mask_img[out_mask[0]] = int(out_obj_id)

            part_index = len(raw_parts)
            raw_parts.append(mask_img.cpu().numpy().astype(np.uint16).tobytes())
            frame_parts.append({"frame_idx": int(out_frame_idx), "mask_part_index": part_index})

        if not tracked_any:
            return {"ok": False, "reason": "tracking_failed"}, [], None

        meta = {
            "ok": True,
            "reason": "ok",
            "num_frames": len(frame_parts),
            "num_instances": len(inference_state.get("obj_ids", [])),
            "instance_labels": {},
            "mask_dtype": "uint16",
            "mask_shape": [int(video_height), int(video_width)],
            "frame_parts": frame_parts,
        }
        return meta, raw_parts, inference_state

    def _handle_segment_instances(self, message_parts: list[bytes]) -> None:
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

        try:
            pil_images = [
                Image.open(io.BytesIO(part)).convert("RGB")
                for part in message_parts[2:]
            ]
        except Exception as exc:
            self._reply({"ok": False, "reason": f"bad_image_bytes:{exc}"})
            return

        if mode == "segment":
            target_text = str(spec.get("target_text", "")).strip()
            if not target_text:
                self._reply({"ok": False, "reason": "missing_target_text"})
                return

            box_threshold = float(spec.get("box_threshold", 0.25))
            text_threshold = float(spec.get("text_threshold", 0.25))
            # Segment on the first image, then track the rest of the chunk.
            meta, raw_parts, inference_state = self._segment_and_track_chunk(
                target_text=target_text,
                pil_images=pil_images,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                start_frame_idx=0,
            )
            if meta.get("ok") and inference_state is not None:
                self.inference_state_dict[state_key] = inference_state
            meta["state_key"] = state_key
            self._reply(meta, raw_parts)
        elif mode == "track":
            existing_state = self.inference_state_dict.get(state_key)
            if existing_state is None:
                self._reply({"ok": False, "reason": "state_not_found", "state_key": state_key})
                return
            meta, raw_parts, updated_state = self._track_chunk_from_state(
                inference_state=existing_state,
                pil_images=pil_images,
            )
            if meta.get("ok") and updated_state is not None:
                self.inference_state_dict[state_key] = updated_state
            meta["state_key"] = state_key
            self._reply(meta, raw_parts)
        else:
            self._reply({"ok": False, "reason": "invalid_mode", "state_key": state_key})

    def run(self) -> None:
        print(f"GSAM server ready on {self.endpoint} (REP), command=segment_instances")
        while True:
            message_parts = self.socket.recv_multipart(flags=0)
            if not message_parts:
                self._reply({"ok": False, "reason": "empty_request"})
                continue

            cmd = message_parts[0]
            if cmd == b"segment_instances":
                self._handle_segment_instances(message_parts)
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
    args = parser.parse_args()

    checkpoint, cfg = SAM2_MODELS[args.model]
    server = ServerGSAM(endpoint=args.endpoint, sam2_checkpoint=checkpoint, model_cfg=cfg)
    server.run()
