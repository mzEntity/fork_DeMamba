"""Run with: python -m unittest discover -s tests -v."""
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import torch

from dataloader import GenVideo
from dataloader.AIGVDBench import AIGVDBenchDataset, _read_video_frames
from dataloader.video_processing import (
    crop_center_by_percentage, prepare_clip, sample_frame_indices, virtual_frame_indices,
)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


class SamplingTests(unittest.TestCase):
    def test_long_video_uses_two_fps_not_adjacent_original_frames(self):
        self.assertEqual(virtual_frame_indices(300, 30), list(range(0, 300, 15)))
        seed(7)
        positions = sample_frame_indices(20)
        self.assertEqual(len(positions), 8)
        self.assertEqual(positions, list(range(positions[0], positions[0] + 8)))
        self.assertGreater(positions[0], 0)
        self.assertEqual(virtual_frame_indices(120, 30), list(range(0, 120, 15)))

    def test_short_videos_use_duration_dependent_rate(self):
        # 3 seconds -> ceil(8 / 3) = 3 FPS -> 9 candidates.
        self.assertEqual(virtual_frame_indices(90, 30), list(range(0, 90, 10)))
        self.assertEqual(virtual_frame_indices(1, 30), [0] * 8)
        # Nearest frame, with half-up rounding.
        self.assertEqual(virtual_frame_indices(10, 5), [0, 1, 3, 4, 5, 6, 8, 9])

    def test_bad_timing_is_rejected(self):
        for fps in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                virtual_frame_indices(10, fps)
        with self.assertRaises(ValueError):
            virtual_frame_indices(0, 30)

    def test_padding_is_zero_after_normalization(self):
        images = [np.full((30, 50, 3), 80 + i, np.uint8) for i in range(3)]
        for training in (True, False):
            seen = []
            def read(index):
                seen.append(index)
                return images[index]
            result = prepare_clip(3, read, training=training)
            self.assertEqual(seen, [0, 1, 2])
            self.assertEqual(tuple(result.shape), (8, 3, 224, 224))
            self.assertEqual(result.dtype, torch.float32)
            self.assertEqual(torch.count_nonzero(result[3:]).item(), 0)
            self.assertGreater(torch.count_nonzero(result[:3]).item(), 0)

    def test_crop_and_eval_normalization(self):
        for height, width in ((20, 40), (40, 20), (20, 20)):
            image = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
            if width > height:
                expected = image[:, int(width * .1):width - int(width * .1)]
            else:
                expected = image[int(height * .1):height - int(height * .1)]
            np.testing.assert_array_equal(crop_center_by_percentage(image, .1), expected)
            expected = cv2.resize(expected, (224, 224)).astype(np.float32) / 255
            expected = (expected - np.array([.485, .456, .406])) / np.array([.229, .224, .225])
            actual = prepare_clip(8, lambda i: image, training=False)[0]
            np.testing.assert_allclose(actual.numpy(), expected.transpose(2, 0, 1), atol=1e-6)


class DatasetParityTests(unittest.TestCase):
    def test_image_list_and_direct_video_paths_match(self):
        # A lossless source isolates sampling/processing from permitted JPEG loss.
        with tempfile.TemporaryDirectory() as directory:
            for width, height, fps, count in ((64, 32, 10, 60), (32, 64, 12, 18), (32, 32, 5, 1)):
                path = str(Path(directory) / f'{width}_{height}_{count}.avi')
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'FFV1'), fps, (width, height))
                self.assertTrue(writer.isOpened())
                rng = np.random.default_rng(5)
                rgb_frames = []
                for i in range(count):
                    rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
                    rgb_frames.append(rgb)
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                writer.release()
                # Independently materialize the approximate extracted-frame sequence.
                duration = count / fps
                target_fps = 2 if duration >= 4 else int(np.ceil(8 / duration))
                timestamps = np.arange(0, duration - 1e-10, 1 / target_fps)
                reference = [rgb_frames[min(count - 1, int(t * fps + .5))] for t in timestamps]
                for training in (False, True):
                    for source in ('fake/test', 'real/youku'):
                        label = int(source.startswith('fake'))
                        row = {'content_path': source, 'label': label,
                               'frame_seq': repr(list(range(1, len(reference) + 1))), 'image_path': ''}
                        df = pd.DataFrame([row, dict(row, label=1-label)])
                        if training:
                            legacy = GenVideo.Ours_Dataset_train([0, 1], df)
                            legacy.balanced_indices = [0, 1]
                        else:
                            legacy = GenVideo.Ours_Dataset_val({'task': 'normal'}, [0], df)
                        direct = AIGVDBenchDataset(
                            [{'path': path, 'label': label, 'source': source, 'video_id': 'test/' + source}],
                            training=training,
                        )
                        def read_image(image_path):
                            return reference[int(Path(image_path).stem) - 1].copy()
                        # GenVideo's OSS callback is deployment-specific and absent here.
                        with patch.object(GenVideo, 'download_oss_file', read_image, create=True):
                            for value in (0, 1, 2, 3, 4, 5):
                                seed(value)
                                expected = legacy[0]
                                seed(value)
                                actual = direct[0]
                                torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                                torch.testing.assert_close(actual[2], expected[2])
                                torch.testing.assert_close(actual[3], expected[3])
                                self.assertEqual(len(actual), 4 if training else 5)

    def test_decode_failure_releases_capture(self):
        with patch('dataloader.AIGVDBench.cv2.VideoCapture') as factory:
            capture = factory.return_value
            capture.isOpened.return_value = True
            capture.get.side_effect = [100, 25]
            capture.read.return_value = (False, None)
            with self.assertRaisesRegex(RuntimeError, 'Could not decode frame'):
                _read_video_frames('bad.mp4', False)
            capture.release.assert_called_once()


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
