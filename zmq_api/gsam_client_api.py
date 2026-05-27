import argparse
import io
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import zmq


class GSAMClientAPI:
    def __init__(
        self,
        server_endpoint: str = "tcp://127.0.0.1:8091",
    ) -> None:
        self.server_endpoint = server_endpoint

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.server_endpoint)

    def close(self) -> None:
        self.socket.close(0)
        self.context.term()

    def _send_command(
        self,
        *,
        command: bytes,
        spec: Optional[Dict[str, Any]] = None,
        image_parts: Optional[List[bytes]] = None,
    ) -> List[bytes]:
        payload: List[bytes] = [command]
        if spec is not None:
            payload.append(json.dumps(spec).encode("utf-8"))
        if image_parts:
            payload.extend(image_parts)

        self.socket.send_multipart(payload)
        return self.socket.recv_multipart()

    def _encode_pil_image_as_jpeg_bytes(self, image: Image.Image) -> bytes:
        img = image.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue()

    def _parse_meta(self, parts: List[bytes]) -> Dict[str, Any]:
        if not parts:
            return {"ok": False, "reason": "empty_reply"}
        try:
            return json.loads(parts[0].decode("utf-8"))
        except Exception as exc:
            return {"ok": False, "reason": f"bad_reply_json:{exc}"}

    @staticmethod
    def _normalize_instance_confidences(value: Any) -> Dict[int, Dict[str, Any]]:
        if not isinstance(value, dict):
            return {}

        normalized: Dict[int, Dict[str, Any]] = {}
        for raw_instance_id, info in value.items():
            try:
                instance_id = int(raw_instance_id)
            except (TypeError, ValueError):
                continue

            if isinstance(info, dict):
                normalized[instance_id] = dict(info)
                continue

            try:
                normalized[instance_id] = {"confidence": float(info)}
            except (TypeError, ValueError):
                continue
        return normalized

    def _decode_masks_from_reply(
        self, parts: List[bytes], meta: Dict[str, Any]
    ) -> List[np.ndarray]:
        mask_dtype = np.dtype(meta.get("mask_dtype", "uint16"))
        mask_shape = tuple(meta.get("mask_shape", []))
        if len(mask_shape) != 2:
            return []

        frame_parts = meta.get("frame_parts", [])
        ordered_frame_parts = sorted(
            frame_parts, key=lambda fp: int(fp.get("mask_part_index", -1))
        )

        masks: List[np.ndarray] = []
        for frame_info in ordered_frame_parts:
            part_index = int(frame_info["mask_part_index"])
            payload_index = part_index + 1
            if payload_index >= len(parts):
                continue
            mask = np.frombuffer(parts[payload_index], dtype=mask_dtype).reshape(
                mask_shape
            )
            masks.append(mask)
        return masks

    def segment_instances(
        self,
        *,
        images: List[Image.Image],
        mode: str = "segment",
        target_text: Optional[str] = None,
        state_key: Optional[str] = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> Tuple[Dict[str, Any], List[np.ndarray]]:
        normalized_mode = (mode or "").strip().lower()
        if normalized_mode not in {"segment", "track"}:
            return {"ok": False, "reason": "invalid_mode"}, []

        if not images:
            return {"ok": False, "reason": "no_images"}, []

        spec: Dict[str, Any] = {"mode": normalized_mode}
        if state_key:
            spec["state_key"] = state_key

        if normalized_mode == "segment":
            text = (target_text or "").strip()
            if not text:
                return {"ok": False, "reason": "missing_target_text"}, []
            spec["target_text"] = text
            spec["box_threshold"] = float(box_threshold)
            spec["text_threshold"] = float(text_threshold)

        image_parts = [self._encode_pil_image_as_jpeg_bytes(img) for img in images]
        parts = self._send_command(
            command=b"segment_instances", spec=spec, image_parts=image_parts
        )
        meta = self._parse_meta(parts)
        if not meta.get("ok"):
            return meta, []

        meta["instance_confidences"] = self._normalize_instance_confidences(
            meta.get("instance_confidences")
        )

        masks = self._decode_masks_from_reply(parts, meta)
        if not masks and int(meta.get("num_frames", 0)) > 0:
            return {"ok": False, "reason": "bad_mask_payload"}, []

        return meta, masks

    def copy_state(
        self, *, source_state_key: str, new_state_key: Optional[str] = None
    ) -> Dict[str, Any]:
        spec: Dict[str, Any] = {"source_state_key": source_state_key}
        if new_state_key:
            spec["new_state_key"] = new_state_key
        parts = self._send_command(command=b"copy_state", spec=spec)
        return self._parse_meta(parts)

    def remove_state(self, *, state_key: str) -> Dict[str, Any]:
        parts = self._send_command(
            command=b"remove_state", spec={"state_key": state_key}
        )
        return self._parse_meta(parts)

    def remove_all_states(self) -> Dict[str, Any]:
        parts = self._send_command(command=b"remove_all_states")
        return self._parse_meta(parts)

    def list_states(self) -> Dict[str, Any]:
        """Return server state summaries keyed by state_key.

        Each state summary includes only: video_height, video_width, and num_frames.
        """
        parts = self._send_command(command=b"list_states")
        return self._parse_meta(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="ClientGSAM API smoke check")
    parser.add_argument(
        "--server-endpoint", default="tcp://127.0.0.1:8091", help="ZMQ server endpoint"
    )
    args = parser.parse_args()

    client = GSAMClientAPI(server_endpoint=args.server_endpoint)
    print(f"Connected ClientGSAM to {args.server_endpoint}")
    client.close()


if __name__ == "__main__":
    main()
