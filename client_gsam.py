import argparse
import io
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import supervision as sv
import zmq


class ClientGSAM:
    def __init__(
        self,
        server_endpoint: str = "tcp://127.0.0.1:8091",
        dataset_dir: str = "client_demo/dataset",
        output_dir: str = "client_demo/output",
        chunk_size: int = 20,
        target_text: str = "object.",
    ) -> None:
        self.server_endpoint = server_endpoint
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.target_text = self._normalize_target_text(target_text)

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.server_endpoint)
        self.mask_annotator = sv.MaskAnnotator(opacity=0.45)
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.instance_labels: dict[int, str] = {}

    def _normalize_target_text(self, text: str) -> str:
        t = (text or "").strip().lower()
        if not t.endswith("."):
            t = t + "."
        return t

    def _clear_output_dir(self) -> None:
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _collect_image_paths(self) -> list[Path]:
        if not self.dataset_dir.exists():
            return []

        image_paths = [
            p
            for p in self.dataset_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        def _sort_key(path: Path):
            stem = path.stem
            return (0, int(stem)) if stem.isdigit() else (1, stem)

        image_paths.sort(key=_sort_key)
        return image_paths

    def _chunk_paths(self, image_paths: list[Path]) -> list[list[Path]]:
        return [
            image_paths[i : i + self.chunk_size]
            for i in range(0, len(image_paths), self.chunk_size)
        ]

    def _encode_image_as_jpeg_bytes(self, image_path: Path) -> bytes:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=92)
            return buf.getvalue()

    def _send_segment_instances(
        self,
        *,
        mode: str,
        chunk_paths: list[Path],
        state_key: str | None,
    ) -> tuple[dict[str, Any], list[np.ndarray]]:
        spec: dict[str, Any] = {"mode": mode}
        if mode == "segment":
            spec["target_text"] = self.target_text
        if state_key:
            spec["state_key"] = state_key

        image_parts = [self._encode_image_as_jpeg_bytes(p) for p in chunk_paths]
        self.socket.send_multipart(
            [b"segment_instances", json.dumps(spec).encode("utf-8"), *image_parts]
        )

        parts = self.socket.recv_multipart()
        meta = json.loads(parts[0].decode("utf-8"))
        if not meta.get("ok"):
            return meta, []

        mask_dtype = np.dtype(meta.get("mask_dtype", "uint16"))
        mask_shape = tuple(meta.get("mask_shape", []))
        if len(mask_shape) != 2:
            return {"ok": False, "reason": "bad_mask_shape"}, []

        frame_parts = meta.get("frame_parts", [])
        ordered_frame_parts = sorted(
            frame_parts, key=lambda fp: int(fp.get("mask_part_index", -1))
        )

        masks: list[np.ndarray] = []
        for frame_info in ordered_frame_parts:
            part_index = int(frame_info["mask_part_index"])
            payload_index = part_index + 1
            if payload_index >= len(parts):
                continue
            mask = np.frombuffer(parts[payload_index], dtype=mask_dtype).reshape(mask_shape)
            masks.append(mask)

        return meta, masks

    def _update_instance_labels(self, meta: dict[str, Any]) -> None:
        raw_labels = meta.get("instance_labels", {})
        if not isinstance(raw_labels, dict):
            return

        for raw_obj_id, raw_label in raw_labels.items():
            try:
                obj_id = int(raw_obj_id)
            except (TypeError, ValueError):
                continue
            self.instance_labels[obj_id] = str(raw_label)

    def _render_and_save_chunk(self, chunk_paths: list[Path], masks: list[np.ndarray]) -> None:
        # Server returns masks in chunk order. If counts differ, render overlap safely.
        for idx, image_path in enumerate(chunk_paths):
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue

            if idx < len(masks):
                mask_u16 = masks[idx]
                object_ids = [int(x) for x in np.unique(mask_u16) if int(x) > 0]

                if object_ids:
                    instance_masks = np.stack(
                        [(mask_u16 == obj_id) for obj_id in object_ids], axis=0
                    )
                    detections = sv.Detections(
                        xyxy=sv.mask_to_xyxy(instance_masks),
                        mask=instance_masks,
                        class_id=np.array(object_ids, dtype=np.int32),
                    )
                    frame = self.mask_annotator.annotate(scene=frame, detections=detections)
                    frame = self.box_annotator.annotate(scene=frame, detections=detections)
                    labels = [
                        self.instance_labels.get(obj_id, f"id:{obj_id}")
                        for obj_id in object_ids
                    ]
                    frame = self.label_annotator.annotate(
                        scene=frame, detections=detections, labels=labels
                    )

            out_path = self.output_dir / image_path.name
            cv2.imwrite(str(out_path), frame)

    def run(self) -> None:
        self._clear_output_dir()

        image_paths = self._collect_image_paths()
        if not image_paths:
            print(f"No images found in {self.dataset_dir}")
            return

        chunks = self._chunk_paths(image_paths)
        state_key: str | None = None

        for chunk_idx, chunk_paths in enumerate(chunks):
            mode = "segment" if chunk_idx == 0 else "track"
            if mode == "track" and not state_key:
                print("Missing state_key for track mode; stopping.")
                break

            meta, masks = self._send_segment_instances(
                mode=mode,
                chunk_paths=chunk_paths,
                state_key=state_key,
            )

            if not meta.get("ok"):
                print(f"Chunk {chunk_idx} failed: {meta.get('reason')}")
                # Save original frames even when server fails.
                self._render_and_save_chunk(chunk_paths, [])
                continue

            self._update_instance_labels(meta)
            state_key = str(meta.get("state_key", state_key or "")) or state_key
            self._render_and_save_chunk(chunk_paths, masks)
            print(
                f"Chunk {chunk_idx} ok | mode={mode} | state_key={state_key} | "
                f"frames={len(chunk_paths)} | masks={len(masks)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="GSAM chunked client")
    parser.add_argument(
        "--server-endpoint", default="tcp://127.0.0.1:8091", help="ZMQ server endpoint"
    )
    parser.add_argument(
        "--dataset-dir", default="client_demo/dataset", help="Input image directory"
    )
    parser.add_argument(
        "--output-dir", default="client_demo/output", help="Output image directory"
    )
    parser.add_argument("--chunk-size", type=int, default=20, help="Chunk size")
    parser.add_argument("--target-text", default="object.", help="Grounding text prompt")

    args = parser.parse_args()

    client = ClientGSAM(
        server_endpoint=args.server_endpoint,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        target_text=args.target_text,
    )
    client.run()


if __name__ == "__main__":
    main()
