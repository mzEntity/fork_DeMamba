import importlib.util
from pathlib import Path
import random
import tempfile
import unittest

import cv2
import numpy as np
import torch

from dataloader.AIGVDBench import AIGVDBenchDataset, _sample_indices


spec = importlib.util.spec_from_file_location(
    'temporal', Path(__file__).resolve().parents[1] / 'models/temporal.py'
)
temporal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(temporal)


class TemporalTests(unittest.TestCase):
    def test_spatial_identity_across_time_and_batches(self):
        tokens = torch.arange(2 * 8 * 196, dtype=torch.float32).reshape(2, 8, 196, 1)
        tokens.requires_grad_()
        blocks = temporal.temporal_blocks(tokens, 1)
        for b in range(2):
            for p in range(196):
                torch.testing.assert_close(blocks[b * 196 + p, :, 0], tokens[b, :, p])
        blocks.sum().backward()
        torch.testing.assert_close(tokens.grad, torch.ones_like(tokens))

    def test_two_by_two_blocks_follow_snake(self):
        tokens = torch.arange(2 * 3 * 16).reshape(2, 3, 16, 1)
        blocks = temporal.temporal_blocks(tokens, 2)
        # Independent coordinate reference: each contiguous 2x2 region,
        # traversed down column 0 then up column 1, in every frame.
        for b in range(2):
            for row in range(2):
                for col in range(2):
                    positions = [(2*row, 2*col), (2*row+1, 2*col),
                                 (2*row+1, 2*col+1), (2*row, 2*col+1)]
                    expected = tokens[b, :, [r*4+c for r, c in positions]]
                    torch.testing.assert_close(blocks[b*4+row*2+col], expected)

    def test_invalid_grid(self):
        with self.assertRaises(ValueError):
            temporal.temporal_blocks(torch.zeros(1, 8, 15, 1), 1)


class SamplingTests(unittest.TestCase):
    def test_clip_coverage_and_short_clips(self):
        random.seed(42)
        for length in (1, 3, 8, 37, 240):
            for training in (False, True):
                indices = _sample_indices(length, 8, training)
                self.assertEqual(len(indices), 8)
                self.assertEqual(indices, sorted(indices))
                self.assertTrue(all(0 <= i < length for i in indices))
                if length >= 8:
                    for bin_id, index in enumerate(indices):
                        self.assertTrue(bin_id*length//8 <= index < (bin_id+1)*length//8)
            self.assertEqual(_sample_indices(length, 8, False), _sample_indices(length, 8, False))

    def test_video_tensor_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'clip.avi')
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 24, (32, 32))
            self.assertTrue(writer.isOpened())
            for i in range(16):
                writer.write(np.full((32, 32, 3), i * 10, dtype=np.uint8))
            writer.release()
            for label in (0, 1):
                records = [{'path': path, 'label': label, 'video_id': 'test/clip'}]
                for training in (False, True):
                    result = AIGVDBenchDataset(records, training, augment=True)[0]
                    self.assertEqual(result[1].shape, (1, 8, 3, 224, 224))
                    self.assertTrue(torch.isfinite(result[1]).all())
                    self.assertEqual(result[2].argmax().item(), label)
                    self.assertEqual(result[3].item(), label)
                    self.assertEqual(len(result), 4 if training else 5)


if __name__ == '__main__':
    unittest.main()
