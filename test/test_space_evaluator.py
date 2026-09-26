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
            support_coverage_min_known_fraction=0.0,
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

    def test_hard_traversal_inflation_is_independent(self):
        evaluator = self.evaluator()
        evaluator.evidence[(2, 0, 1)] = 3
        self.assertNotEqual(
            evaluator.column_state(0.0, 0.0, 0.0, inflation_m=0.4), OCCUPIED
        )
        self.assertEqual(
            evaluator.column_state(0.0, 0.0, 0.0, inflation_m=1.0), OCCUPIED
        )

    def test_subvoxel_footprint_still_inflates_adjacent_occupancy(self):
        evaluator = self.evaluator()
        evaluator.evidence[(1, 0, 1)] = 3
        self.assertEqual(
            evaluator.column_state(0.0, 0.0, 0.0, inflation_m=0.45), OCCUPIED
        )

    def test_diagonal_unknown_direction_is_supported(self):
        evaluator = self.evaluator(frontier_lateral_m=0.0)
        result = evaluator.direction_evidence(0.0, 0.0, 0.0, 1, 1, 0.5)
        self.assertTrue(result.is_frontier)
        self.assertGreaterEqual(result.unknown_columns, 2)

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

    def test_registered_view_covers_only_visible_support_in_range(self):
        evaluator = self.evaluator(
            support_coverage_range_m=4.0,
        )
        added = evaluator.observe_support_cells(
            (0.0, 0.0, 1.0),
            ((3.0, 0.0, 0.0), (6.0, 0.0, 0.0)),
            1.0,
        )
        self.assertEqual(added, 1)
        self.assertTrue(evaluator.is_support_covered(3.0, 0.0, 0.0))
        self.assertFalse(evaluator.is_support_covered(6.0, 0.0, 0.0))

    def test_obstacle_occludes_support_coverage(self):
        evaluator = self.evaluator(
            support_coverage_range_m=5.0,
        )
        evaluator.evidence[(2, 0, 1)] = 3
        evaluator._column_cache.clear()
        evaluator.observe_support_cells(
            (0.0, 0.0, 1.0), ((3.0, 0.0, 0.0),), 1.0
        )
        self.assertFalse(evaluator.is_support_covered(3.0, 0.0, 0.0))

    def test_unknown_line_is_not_counted_as_scan_coverage(self):
        evaluator = self.evaluator(
            support_coverage_range_m=5.0,
            support_coverage_min_known_fraction=0.6,
        )
        evaluator.observe_support_cells(
            (0.0, 0.0, 1.0), ((3.0, 0.0, 0.0),), 1.0
        )
        self.assertFalse(evaluator.is_support_covered(3.0, 0.0, 0.0))
        for ix in range(1, 6):
            evaluator.evidence[(ix, 0, 1)] = -2
        evaluator._column_cache.clear()
        evaluator.observe_support_cells(
            (0.0, 0.0, 1.0), ((3.0, 0.0, 0.0),), 1.0
        )
        self.assertTrue(evaluator.is_support_covered(3.0, 0.0, 0.0))

    def test_prediction_can_require_partial_known_line(self):
        evaluator = self.evaluator(support_coverage_range_m=5.0)
        point = (3.0, 0.0, 0.0)
        self.assertEqual(
            evaluator.support_visibility_keys(
                (0.0, 0.0, 0.0),
                (point,),
                1.0,
                minimum_known_fraction=0.5,
            ),
            (),
        )
        for ix in range(1, 6):
            evaluator.evidence[(ix, 0, 1)] = -2
        evaluator._column_cache.clear()
        self.assertEqual(
            len(
                evaluator.support_visibility_keys(
                    (0.0, 0.0, 0.0),
                    (point,),
                    1.0,
                    minimum_known_fraction=0.5,
                )
            ),
            1,
        )

    def test_line_of_sight_accepts_frontier_footprint_override(self):
        evaluator = self.evaluator()
        evaluator.evidence[(1, 1, 1)] = 3
        self.assertTrue(
            evaluator.has_clear_line_of_sight(
                (0.0, 0.0, 0.0), (2.0, 0.0, 0.0), inflation_m=0.0
            )
        )
        self.assertFalse(
            evaluator.has_clear_line_of_sight(
                (0.0, 0.0, 0.0), (2.0, 0.0, 0.0), inflation_m=0.45
            )
        )

    def test_clearance_uses_nearest_obstacle_column(self):
        evaluator = self.evaluator(clearance_search_m=3.0)
        evaluator.evidence[(4, 0, 1)] = 3
        evaluator._clearance_cache.clear()
        self.assertAlmostEqual(
            evaluator.obstacle_clearance(0.0, 0.0, 0.0), 1.75
        )

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
