"""Export per-video AIGVDBench test predictions without computing metrics."""

import argparse
import csv
from pathlib import Path

import cv2
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import util
from dataloader.AIGVDBench import AIGVDBenchDataset, _collect_records

# OpenCV builds differ; silence decoder logs when this API is available.
_set_cv_log_level = getattr(cv2, "setLogLevel", None)
if _set_cv_log_level is not None:
    _set_cv_log_level(0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Inference/test YAML configuration")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if cfg.get("dataset") != "AIGVDBench":
        raise ValueError("This exporter requires dataset: AIGVDBench")

    checkpoint_paths = cfg.get("checkpoints")
    if not checkpoint_paths or not isinstance(checkpoint_paths, list):
        raise ValueError("Test configuration must define a non-empty 'checkpoints' list")

    configured_device = cfg.get("test_device", "auto")
    if configured_device == "auto":
        configured_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(configured_device)
    records = _collect_records(
        cfg["aigvdbench_data_root"], cfg["aigvdbench_meta_root"], cfg.get("aigvdbench_test_split", "test"), "all"
    )
    loader = DataLoader(
        AIGVDBenchDataset(records, training=False),
        batch_size=int(cfg.get("test_batch_size", 8)),
        shuffle=False,
        num_workers=int(cfg.get("test_num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    # Keep each checkpoint's model resident so every decoded video batch is
    # shared across checkpoints. This trades device memory for avoiding repeat
    # CPU video decoding.
    models = []
    for checkpoint_path in checkpoint_paths:
        checkpoint_path = Path(checkpoint_path)
        model = util.build_model(cfg).to(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        try:
            model.load_state_dict(state)
        except RuntimeError:
            # Training wraps the model in DataParallel, which prefixes keys with "module.".
            if state and all(key.startswith("module.") for key in state):
                model.load_state_dict({key[len("module."):]: value for key, value in state.items()})
            else:
                raise
        model.eval()
        models.append(model)

    outputs = []
    output_dir = Path(cfg.get("test_output_dir", "./results"))
    for checkpoint_path in checkpoint_paths:
        checkpoint_stem = Path(checkpoint_path).stem
        output = output_dir / f"test_predictions_{checkpoint_stem}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        outputs.append(output)
    rows_by_checkpoint = [[] for _ in models]
    skipped_batches = []
    with torch.inference_mode():
        loader_iterator = iter(loader)
        progress = tqdm(total=len(loader), desc="Testing AIGVDBench", unit="batch")
        for batch_index in range(len(loader)):
            try:
                _, video, onehot_labels, _, video_ids = next(loader_iterator)
            except StopIteration:
                break
            except RuntimeError as exc:
                # A decode error raised inside Dataset.__getitem__ is wrapped
                # by DataLoader when workers are enabled. Drop that full batch
                # and keep consuming the remaining batches.
                error = str(exc)
                if "DataLoader worker process" not in error and "Could not decode frame" not in error:
                    raise
                skipped_batches.append((batch_index + 1, error))
                progress.update(1)
                progress.set_postfix(skipped=len(skipped_batches), status="batch skipped")
                tqdm.write(f"Skipped batch {batch_index + 1}/{len(loader)} due to video decode error: {error}")
                continue

            inputs = video[:, 0].float().to(device, non_blocking=device.type == "cuda")
            actual_labels = onehot_labels.argmax(dim=1).tolist()
            for model_index, model in enumerate(models):
                progress.set_postfix(checkpoint=Path(checkpoint_paths[model_index]).stem)
                fake_scores = model(inputs)[:, 0].sigmoid().cpu().tolist()
                rows = rows_by_checkpoint[model_index]
                for video_id, fake_score, actual_label in zip(video_ids, fake_scores, actual_labels):
                    path_parts = str(video_id).split("/")
                    if len(path_parts) < 3:
                        raise ValueError(f"Unexpected AIGVDBench video id: {video_id}")
                    source = "/".join(path_parts[1:-1])
                    filename = path_parts[-1]
                    real_score = 1.0 - fake_score
                    predicted = "fake" if fake_score >= 0.5 else "real"
                    actual = "fake" if actual_label == 1 else "real"
                    rows.append((source, filename, real_score, fake_score, predicted, actual))
            progress.update(1)
            progress.set_postfix(skipped=len(skipped_batches))
        progress.close()

    for rows, output in zip(rows_by_checkpoint, outputs):
        with output.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(("source", "video_name", "real_score", "fake_score", "predicted_result", "actual_result"))
            writer.writerows(rows)
        print(f"Exported {len(rows)} video predictions to {output}")

    if skipped_batches:
        error_log = output_dir / "test_skipped_batches.txt"
        with error_log.open("w", encoding="utf-8") as stream:
            for batch_index, error in skipped_batches:
                stream.write(f"batch {batch_index}: {error}\n\n")
        print(f"Skipped {len(skipped_batches)} batch(es); details saved to {error_log}")


if __name__ == "__main__":
    main()
