"""Integration coverage for merged loader, training, reporting and checkpoints."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
import yaml

import util
from dataloader import generate_dataset_loader


class SmallVideoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 1)

    def forward(self, video):
        return self.head(video.mean(dim=(1, 3, 4)))


class TrainingPipelineTests(unittest.TestCase):
    def make_loaders(self, directory):
        root = Path(directory)
        cfg_path = Path(__file__).resolve().parents[1] / 'configs/XCLIP_DeMamba_AIGVDBench.yaml'
        cfg = yaml.safe_load(cfg_path.read_text())
        cfg.update(aigvdbench_data_root=str(root / 'videos'),
                   aigvdbench_meta_root=str(root / 'meta'), num_workers=0,
                   train_batch_size=2, val_batch_size=2)
        for split in ('train', 'test'):
            for source, label in (('real/real', 0), ('fake/T2V_Open-Sora', 1)):
                path = root / 'videos' / split / source / 'clip.avi'
                path.parent.mkdir(parents=True)
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 8, (32, 32))
                self.assertTrue(writer.isOpened())
                for i in range(16):
                    writer.write(np.full((32, 32, 3), 30 + label * 100 + i, np.uint8))
                writer.release()
                meta = root / 'meta' / split / (source + '.json')
                meta.parent.mkdir(parents=True, exist_ok=True)
                meta.write_text(json.dumps({'entries': [{'filename': 'clip.avi', 'label': label}]}))
        train, val = generate_dataset_loader(cfg)
        return cfg, train, val

    def test_train_validate_report_and_save_without_best_selection(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            cfg, train, val = self.make_loaders(directory)
            # Irrelevant checkpoint-source restrictions must not block monitoring.
            cfg['aigvdbench_checkpoint_sources'] = ['fake/not-in-test']
            model = SmallVideoModel()
            before = model.head.weight.detach().clone()
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            util.train_one_epoch(cfg, model, torch.nn.BCEWithLogitsLoss(), scheduler,
                                 optimizer, 0, 0, -1, train, val, directory)
            self.assertFalse(torch.equal(before, model.head.weight.detach()))
            self.assertFalse((Path(directory) / 'best_acc.pth').exists())
            for name in ('last.pth', 'Epoch_1.pth'):
                saved = torch.load(Path(directory) / name, weights_only=False)
                self.assertEqual(saved['pipeline_version'], util.PIPELINE_VERSION)
                self.assertEqual(saved['epoch'], 1)
            report = (Path(directory) / 'Epoch_1_accuracy.txt').read_text()
            for metric in ('AUC:', 'ACC@5%FPR:', 'threshold:', 'FPR:', '来源:'):
                self.assertIn(metric, report)

    def test_best_checkpoint_uses_selected_sources_not_overall_accuracy(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            cfg, train, val = self.make_loaders(directory)
            cfg.update(save_best_checkpoint=True,
                       aigvdbench_checkpoint_sources=['real/real', 'fake/T2V_Open-Sora'])
            model = SmallVideoModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            evaluation = (2/3, ['test/real/real/a', 'test/fake/T2V_Open-Sora/b', 'test/fake/other/c'],
                          [0, 1, 0], np.array([0, 1, 1]), np.array([.1, .9, .2]))
            with patch.object(util, 'eval_model', return_value=evaluation):
                _, best, _ = util.train_one_epoch(
                    cfg, model, torch.nn.BCEWithLogitsLoss(), scheduler, optimizer,
                    0, 0, .9, train, val, directory)
            self.assertEqual(best, 1)
            self.assertTrue((Path(directory) / 'best_acc.pth').exists())



if __name__ == '__main__':
    unittest.main()
