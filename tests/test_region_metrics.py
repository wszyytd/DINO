"""No torch: preserve teacher misses, ambiguity and extraction failures in denominators."""
import unittest

from dinoexp.region_ab_metrics import metric_counts
from dinoexp.region_ab_report import aggregate


class RegionMetricsTests(unittest.TestCase):
    def test_null_does_not_rescue_visible_teacher_miss(self):
        labels = {'miss': dict(physical_visible=True, positives=[], candidates=[])}
        predictions = [dict(id='miss', scores=[0.], choice=None, rank_choice=None, extraction_failure=False)]
        m = metric_counts(predictions, labels)
        self.assertEqual(m['downstream_accuracy'], 1.)
        self.assertEqual(m['end_to_end_accuracy'], 0.)
        self.assertEqual(m['teacher_not_confirmable'], 1)
        self.assertIsNone(m['conditional_accuracy'])

    def test_same_covered_denominator_with_false_null_and_failure(self):
        labels = {str(i): dict(physical_visible=True, positives=[0, 1],
                              candidates=[{'ambiguous': False}]*3) for i in range(3)}
        predictions = [dict(id=str(i), scores=[1., 1., 0., 0.], choice=choice, rank_choice=1, extraction_failure=failed)
                       for i, (choice, failed) in enumerate([(1, False), (None, False), (0, True)])]
        m = metric_counts(predictions, labels)
        self.assertEqual(m['covered'], 3)
        self.assertEqual(m['correct_region'], 1)
        self.assertEqual(m['false_null'], 1)
        self.assertEqual(m['wrong_region'], 1)
        self.assertEqual(m['ranking_correct'], 2)
        self.assertEqual(m['ranking_with_distractors'], 3)
        self.assertEqual(m['end_to_end_accuracy'], 1/3)

    def test_absence_ambiguity_and_rejection(self):
        labels = {str(i): dict(physical_visible=False, positives=[], candidates=[{'ambiguous': True}]) for i in range(2)}
        predictions = [dict(id=str(i), scores=[0., 0.], choice=choice, rank_choice=0, extraction_failure=False)
                       for i, choice in enumerate([0, None])]
        m = metric_counts(predictions, labels)
        self.assertEqual(m['absent_false_alarm'], 1)
        self.assertEqual(m['absent_rejected'], 1)
        self.assertEqual(m['ambiguous_selected'], 1)
        self.assertEqual(m['end_to_end_accuracy'], .5)

    def test_paired_seed_summary_keeps_missing_denominators(self):
        metrics = dict(conditional_accuracy=.5, ranking_accuracy=.75, ranking_with_distractors_accuracy=None,
                       downstream_accuracy=.5, visible_recall=.4, absent_rejection=.6, end_to_end_accuracy=.5)
        run = dict(results=[dict(variant=v, seed=s, splits={'probe': {'metrics': metrics}})
                            for v in ('A', 'B') for s in (7, 17, 27)])
        result = aggregate(run)['probe']
        self.assertEqual(result['conditional_accuracy']['B_minus_A']['values'], [0., 0., 0.])
        self.assertIsNone(result['ranking_with_distractors_accuracy']['A']['mean'])


if __name__ == '__main__':
    unittest.main()
