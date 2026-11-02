"""Data loader for the on-disk AIGVDBench video dataset."""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


_RECORD_CACHE = {}
NUM_FRAMES = 8
IMAGE_SIZE = 224
IMAGE_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGE_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


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


def _sample_indices(frame_count, count, training):
    """Cover the whole clip in both modes; jitter only within temporal bins."""
    if frame_count < 1 or count < 1:
        raise ValueError("frame_count and count must be positive")
    if frame_count < count:
        return np.linspace(0, frame_count - 1, num=count).round().astype(int).tolist()
    edges = [i * frame_count // count for i in range(count + 1)]
    return [
        random.randrange(start, stop) if training else (start + stop - 1) // 2
        for start, stop in zip(edges[:-1], edges[1:])
    ]


def _read_video_frames(path, training, augment=False):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")

    video_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_frame_count < 1:
        capture.release()
        raise RuntimeError(f"Video has no readable frames: {path}")

    indices = _sample_indices(video_frame_count, NUM_FRAMES, training)
    # Share augmentation parameters across frames to preserve temporal consistency.
    flip = training and augment and random.random() < 0.5
    jpeg_quality = random.randint(75, 100) if training and augment and random.random() < 0.5 else None
    frames = []
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise RuntimeError(f"Could not decode frame {frame_index} from {path}")

        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        if flip:
            frame = cv2.flip(frame, 1)
        if jpeg_quality is not None:
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                capture.release()
                raise RuntimeError(f"Could not augment video: {path}")
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - IMAGE_MEAN) / IMAGE_STD
        frames.append(torch.from_numpy(frame.transpose(2, 0, 1).copy()))

    capture.release()
    return torch.stack(frames, dim=0)


class AIGVDBenchDataset(Dataset):
    """Return the tensor/label contract consumed by train.py and util.py."""

    def __init__(self, records, training, augment=False):
        self.records = records
        self.training = training
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        frames = _read_video_frames(record["path"], self.training, self.augment)
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
        AIGVDBenchDataset(val_records, training=False),
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
