import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader.AIGVDBench import AIGVDBenchDataset
from dataloader.error_handling import FailedVideo, collate_video_samples, video_batches
from test_training_pipeline import SmallVideoModel
import util


class VideoErrorTests(unittest.TestCase):
    def records(self, directory):
        path = Path(directory) / 'good.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 8, (32, 32))
        self.assertTrue(writer.isOpened())
        for i in range(16):
            writer.write(np.full((32, 32, 3), i * 10, np.uint8))
        writer.release()
        bad = Path(directory) / 'broken.mp4'
        bad.write_bytes(b'not a valid video')
        return [
            {'path': str(path), 'video_id': 'test/real/real/good.avi', 'label': 0},
            {'path': str(bad), 'video_id': 'test/fake/kling/broken.mp4', 'label': 1},
            {'path': str(path), 'video_id': 'test/fake/kling/good.avi', 'label': 1},
        ]

    def test_mixed_batch_preserves_good_videos_and_logs_errors(self):
        for workers in (0, 2):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as directory:
                records = self.records(directory)
                loader = DataLoader(AIGVDBenchDataset(records, False), batch_size=3,
                                    num_workers=workers, collate_fn=collate_video_samples)
                stats = {}
                log = Path(directory) / 'errors.jsonl'
                with contextlib.redirect_stdout(io.StringIO()):
                    batches = list(video_batches(loader, log, 'Validation', 3, stats))
                self.assertEqual(len(batches), 1)
                self.assertEqual(list(batches[0][-1]), [records[0]['video_id'], records[2]['video_id']])
                self.assertEqual(stats['skipped_videos'], 1)
                self.assertEqual(stats['valid_videos'], 2)
                event = json.loads(log.read_text())
                self.assertEqual(event['path'], records[1]['path'])
                self.assertEqual(event['epoch'], 3)
                self.assertIn('could not open', event['reason'])
                self.assertIn('Traceback', event['traceback'])

    def test_arbitrary_preprocessing_exceptions_become_failed_video(self):
        record = {'path': 'a.mp4', 'video_id': 'test/fake/kling/a.mp4', 'label': 1}
        for exception in (ValueError('bad shape'), KeyError('missing field'), OSError('read failed')):
            with patch('dataloader.AIGVDBench._read_video_frames', side_effect=exception):
                failure = AIGVDBenchDataset([record], True)[0]
            self.assertIsInstance(failure, FailedVideo)
            self.assertEqual(failure.error_type, type(exception).__name__)
            self.assertIn(str(exception), failure.reason)

    def test_all_bad_batch_does_not_discard_next_good_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            good, bad, _ = self.records(directory)
            loader = DataLoader(AIGVDBenchDataset([bad, bad, good], False), batch_size=2,
                                collate_fn=collate_video_samples)
            stats = {}
            with contextlib.redirect_stdout(io.StringIO()):
                batches = list(video_batches(loader, Path(directory) / 'errors.jsonl', 'Validation', stats=stats))
            self.assertEqual(len(batches), 1)
            self.assertEqual(list(batches[0][-1]), [good['video_id']])
            self.assertEqual(stats['skipped_videos'], 2)

    def test_completely_invalid_epoch_still_saves_and_reports_nan(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            bad = self.records(directory)[1]
            train = DataLoader(AIGVDBenchDataset([bad], True), collate_fn=collate_video_samples)
            val = DataLoader(AIGVDBenchDataset([bad], False), collate_fn=collate_video_samples)
            model = SmallVideoModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            cfg = {'dataset': 'AIGVDBench', 'val_batch_size': 1, 'save_best_checkpoint': False}
            util.train_one_epoch(cfg, model, torch.nn.BCEWithLogitsLoss(), scheduler, optimizer,
                                 0, 0, -1, train, val, directory)
            self.assertTrue((Path(directory) / 'Epoch_1.pth').exists())
            text = (Path(directory) / 'Epoch_1_accuracy.txt').read_text()
            self.assertIn('AUC: nan%', text)
            self.assertIn("'skipped_videos': 1", text)
            self.assertEqual(scheduler.last_epoch, 0)

    def test_checkpoint_exists_before_validation_starts(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            good = self.records(directory)[0]
            train = DataLoader(AIGVDBenchDataset([good], True), collate_fn=collate_video_samples)
            model = SmallVideoModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            cfg = {'dataset': 'AIGVDBench', 'val_batch_size': 1, 'save_best_checkpoint': False}
            def validation(*args):
                for name in ('last.pth', 'Epoch_1.pth'):
                    self.assertTrue((Path(directory) / name).exists())
                return float('nan'), np.array([], dtype=str), [], np.array([], dtype=int), np.array([])
            with patch.object(util, 'eval_model', side_effect=validation):
                util.train_one_epoch(cfg, model, torch.nn.BCEWithLogitsLoss(), scheduler, optimizer,
                                     0, 0, -1, train, [], directory)


if __name__ == '__main__':
    unittest.main()
