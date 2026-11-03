"""Data loader for the on-disk AIGVDBench video dataset."""

import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Dataset

from .video_processing import prepare_clip, virtual_frame_indices


_RECORD_CACHE = {}


def _collect_records(data_root, meta_root, split, sources):
    """Read one split's metadata and map each entry to its local video file."""
    source_key = tuple(sorted(sources)) if sources != 'all' else 'all'
    cache_key = (str(Path(data_root).resolve()), str(Path(meta_root).resolve()), split, source_key)
    if cache_key in _RECORD_CACHE:
        return _RECORD_CACHE[cache_key]

    split_meta_dir = Path(meta_root) / split
    if not split_meta_dir.is_dir():
        raise FileNotFoundError(f"AIGVDBench metadata split not found: {split_meta_dir}")

    records = []
    missing_videos = 0
    for meta_path in sorted(split_meta_dir.rglob("*.json")):
        relative_group = meta_path.relative_to(split_meta_dir).parent / meta_path.stem
        if sources != 'all' and relative_group.as_posix() not in sources:
            continue
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read metadata file {meta_path}: {exc}") from exc

        entries = metadata["entries"]
        for entry in entries:
            filename = entry["filename"]
            label = entry["label"]
            if not filename or label not in (0, 1):
                continue
            expected_label = {'real': 0, 'fake': 1}.get(relative_group.parts[0])
            if expected_label is not None and label != expected_label:
                raise ValueError(f"Label/source mismatch in {meta_path}: {filename}, label={label}")

            video_path = Path(data_root) / split / relative_group / filename
            if not video_path.is_file():
                missing_videos += 1
                continue
            records.append(
                {
                    "path": str(video_path),
                    "label": int(label),
                    "source": relative_group.as_posix(),
                    "video_id": f"{split}/{relative_group.as_posix()}/{filename}",
                }
            )

    if not records:
        raise RuntimeError(
            f"No videos found for split {split!r}. Check data_root={data_root!r} "
            f"and meta_root={meta_root!r}."
        )
    _RECORD_CACHE[cache_key] = records
    if missing_videos:
        print(f"AIGVDBench {split}: skipped {missing_videos} metadata entries with no MP4 file")
    return records


def _read_video_frames(path, training, video_id='', task='normal'):
    """Emulate GenVideo's extracted JPG sequence without writing images to disk."""
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if frame_count < 1:
            raise RuntimeError(f"Video has no readable frames: {path}")
        try:
            source_indices = virtual_frame_indices(frame_count, fps)
        except ValueError as exc:
            raise RuntimeError(f"Invalid video timing for {path}: {exc}") from exc

        decoded = {}

        def read_frame(index):
            frame_index = source_indices[index]
            if frame_index in decoded:
                return decoded[frame_index].copy()
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not decode frame {frame_index} from {path}")
            decoded[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return decoded[frame_index].copy()

        return prepare_clip(
            len(source_indices), read_frame, training=training,
            video_id=video_id, task=task,
        )
    finally:
        capture.release()


class AIGVDBenchDataset(Dataset):
    """Return the tensor/label contract consumed by train.py and util.py."""

    def __init__(self, records, training, task="normal"):
        self.records = records
        self.training = training
        self.task = task

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        frames = _read_video_frames(
            record["path"], self.training, video_id=record.get("source", ""), task=self.task,
        )
        video = frames.unsqueeze(0)  # [1, T, C, H, W]

        label = record["label"]
        onehot = torch.zeros(2, dtype=torch.float32)
        onehot[label] = 1.0
        binary_label = torch.tensor([float(label)], dtype=torch.float32)
        if self.training:
            return index, video, onehot, binary_label
        return index, video, onehot, binary_label, record["video_id"]


def generate_dataset_loader(cfg):
    data_root = cfg["aigvdbench_data_root"]
    meta_root = cfg["aigvdbench_meta_root"]
    train_split = cfg["aigvdbench_train_split"]
    val_split = cfg["aigvdbench_val_split"]

    train_sources = cfg["aigvdbench_train_sources"]
    val_sources = cfg["aigvdbench_val_sources"]
    train_records = _collect_records(data_root, meta_root, train_split, train_sources)
    if not any(record["label"] == 0 for record in train_records):
        raise RuntimeError("Configured AIGVDBench training sources contain no real videos.")
    if not any(record["label"] == 1 for record in train_records):
        raise RuntimeError("Configured AIGVDBench training sources contain no generated videos.")
    val_records = _collect_records(data_root, meta_root, val_split, val_sources)

    workers = int(cfg["num_workers"])
    train_loader = DataLoader(
        AIGVDBenchDataset(train_records, training=True, augment=cfg.get('train_augment', False)),
        batch_size=int(cfg["train_batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        AIGVDBenchDataset(val_records, training=False, task=cfg.get("task", "normal")),
        batch_size=int(cfg["val_batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )

    print(
        f"AIGVDBench: {len(train_records)} train videos, "
        f"{len(val_records)} validation videos; train split={train_split}, "
        f"validation split={val_split}; train sources={train_sources}"
    )
    return train_loader, val_loader
