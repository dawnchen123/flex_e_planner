#!/usr/bin/env python3

import unittest

import numpy as np

from flex_e_core.space_evaluator import FREE, OCCUPIED, ExplorationSpaceEvaluator


class ExplorationSpaceEvaluatorTest(unittest.TestCase):
    def evaluator(self, **kwargs):
        defaults = dict(
            resolution_m=0.5,
            obstacle_inflation_m=0.0,
            min_clearance_m=0.5,
            max_clearance_m=1.0,
            frontier_lookahead_m=2.0,
            frontier_lateral_m=0.5,
            frontier_min_unknown_columns=2,
        )
        defaults.update(kwargs)
        return ExplorationSpaceEvaluator(**defaults)

    def test_open_unknown_is_frontier_and_wall_is_not(self):
        evaluator = self.evaluator()
        result = evaluator.direction_evidence(0.0, 0.0, 0.0, 1, 0, 0.5)
        self.assertTrue(result.is_frontier)
        evaluator.evidence[(2, 0, 1)] = 3
        evaluator._column_cache.clear()
        self.assertEqual(evaluator.column_state(1.0, 0.0, 0.0), OCCUPIED)
        self.assertFalse(
            evaluator.direction_evidence(0.0, 0.0, 0.0, 1, 0, 0.5).is_frontier
        )

    def test_observed_unsupported_corridor_is_not_frontier(self):
        evaluator = self.evaluator(frontier_lateral_m=0.0, frontier_min_unknown_columns=1)
        for ix in range(1, 5):
            evaluator.evidence[(ix, 0, 1)] = -2
        self.assertTrue(
            all(
                evaluator.column_state(ix * 0.5, 0.0, 0.0) == FREE
                for ix in range(1, 5)
            )
        )
        self.assertFalse(
            evaluator.direction_evidence(0.0, 0.0, 0.0, 1, 0, 0.5).is_frontier
        )

    def test_ray_updates_free_and_occupied_evidence(self):
        evaluator = self.evaluator(min_range_m=0.1, max_range_m=10.0)
        used = evaluator.integrate_scan(
            (0.0, 0.0, 0.5), np.array([[3.0, 0.0, 0.5]])
        )
        self.assertEqual(used, 1)
        self.assertLess(evaluator.evidence[(2, 0, 1)], 0)
        self.assertGreater(evaluator.evidence[(6, 0, 1)], 0)

    def test_completion_mask_is_layer_and_direction_aware(self):
        evaluator = self.evaluator(completed_resolution_m=1.0)
        self.assertGreater(
            evaluator.mark_completed(2.0, 2.0, 0.0, ((1, 0),), 2.0), 0
        )
        self.assertTrue(evaluator.is_completed(2.0, 2.0, 0.0, 1, 0))
        self.assertFalse(evaluator.is_completed(2.0, 2.0, 0.0, -1, 0))
        self.assertFalse(evaluator.is_completed(2.0, 2.0, 3.0, 1, 0))

    def test_closure_requires_complete_untruncated_reachability(self):
        closure = ExplorationSpaceEvaluator.closure_state
        self.assertEqual(closure(0, False, 1000, 100, 30.0, 20.0), "CLOSED")
        self.assertEqual(closure(1, False, 1000, 100, 30.0, 20.0), "OPEN")
        self.assertEqual(
            closure(0, True, 1000, 100, 30.0, 20.0), "SEARCH_TRUNCATED"
        )


if __name__ == "__main__":
    unittest.main()

