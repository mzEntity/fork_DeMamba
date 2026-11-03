import unittest

import numpy as np

from evaluation_metrics import detection_metrics


class DetectionMetricsTests(unittest.TestCase):
    def test_exact_five_percent_and_accuracy(self):
        labels = [0] * 20 + [1, 1]
        scores = np.linspace(.01, .20, 20).tolist() + [.9, .8]
        result = detection_metrics(labels, scores)
        self.assertEqual(result['auc'], 1.0)
        self.assertEqual(result['fpr'], .05)
        self.assertEqual(result['tpr'], 1.0)
        self.assertAlmostEqual(result['threshold'], .20)
        self.assertAlmostEqual(result['acc_at_fpr'], 21 / 22)

    def test_tied_negatives_cannot_be_split_to_reach_five_percent(self):
        labels = [0] * 20 + [1, 1]
        scores = [.8, .8] + [.1] * 18 + [.9, .8]
        result = detection_metrics(labels, scores)
        self.assertEqual(result['fpr'], 0)
        self.assertEqual(result['tpr'], .5)
        self.assertEqual(result['threshold'], .9)
        self.assertAlmostEqual(result['acc_at_fpr'], 21 / 22)

    def test_constant_scores_predict_all_real_at_low_fpr(self):
        result = detection_metrics([0, 0, 1, 1], [.5] * 4)
        self.assertEqual(result['auc'], .5)
        self.assertEqual(result['fpr'], 0)
        self.assertEqual(result['tpr'], 0)
        self.assertEqual(result['acc_at_fpr'], .5)
        self.assertGreater(result['threshold'], .5)

    def test_single_class_metrics_are_undefined(self):
        for labels in ([0, 0], [1, 1]):
            self.assertTrue(all(np.isnan(v) for v in detection_metrics(labels, [.1, .9]).values()))

    def test_acc_is_not_tpr(self):
        result = detection_metrics([0] * 20 + [1, 1], [.1] * 19 + [.9] + [.8, .05])
        self.assertEqual(result['fpr'], .05)
        self.assertEqual(result['tpr'], .5)
        self.assertAlmostEqual(result['acc_at_fpr'], 20 / 22)

    def test_invalid_inputs(self):
        for labels, scores in (([], []), ([0, 1], [.5]), ([0, 2], [.1, .2]),
                               ([0, 1], [.1, float('nan')]), ([0, 1], [-.1, 1.2])):
            with self.assertRaises(ValueError):
                detection_metrics(labels, scores)


if __name__ == '__main__':
    unittest.main()
