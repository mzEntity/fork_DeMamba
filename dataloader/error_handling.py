"""Skip individual failed samples; log in the parent process, never workers."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

from torch.utils.data import default_collate
from tqdm import tqdm


@dataclass
class FailedVideo:
    video_id: str
    path: str
    error_type: str
    reason: str
    traceback: str


def collate_video_samples(samples):
    errors = [asdict(sample) for sample in samples if isinstance(sample, FailedVideo)]
    valid = [sample for sample in samples if not isinstance(sample, FailedVideo)]
    return {'data': default_collate(valid) if valid else None, 'errors': errors}


def video_batches(loader, error_log, stage, epoch=None, stats=None):
    """Accept safe AIGVDBench batches or ordinary batches from other datasets."""
    if stats is None:
        stats = {}
    stats.update(skipped_videos=0, failed_batches=0, valid_videos=0)
    error_log = Path(error_log)
    error_log.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(loader)
    # Appending preserves errors if a run is restarted in the same directory.
    with error_log.open('a', encoding='utf-8') as stream:
        def log(error, batch_index):
            event = dict(error, stage=stage, epoch=epoch, batch=batch_index + 1,
                         timestamp=datetime.now(timezone.utc).isoformat())
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            stream.flush()
            tqdm.write(f"[{stage}] Skipped {event.get('video_id', 'batch')}: "
                       f"{event['error_type']}: {event['reason']}")

        with tqdm(total=len(loader), desc=stage, unit='batch') as progress:
            for batch_index in range(len(loader)):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                except Exception as exc:
                    # Fallback for worker/collation failures outside __getitem__.
                    stats['failed_batches'] += 1
                    log(dict(error_type=type(exc).__name__, reason=str(exc),
                             traceback=traceback.format_exc()), batch_index)
                    progress.update(1)
                    continue
                if isinstance(batch, dict) and 'errors' in batch:
                    for error in batch['errors']:
                        log(error, batch_index)
                    stats['skipped_videos'] += len(batch['errors'])
                    batch = batch['data']
                progress.update(1)
                progress.set_postfix(skipped=stats['skipped_videos'],
                                     failed_batches=stats['failed_batches'])
                if batch is not None:
                    stats['valid_videos'] += len(batch[0])
                    yield batch
                    if 'loss' in stats:
                        progress.set_postfix(loss=stats['loss'], skipped=stats['skipped_videos'],
                                             failed_batches=stats['failed_batches'])
