#!/usr/bin/env python3

import unittest

from flex_e_core.region_coverage import (
    COVERED,
    DEFERRED,
    EXPLORING,
    INACCESSIBLE,
    RegionCoverageTracker,
)


class RegionCoverageTrackerTest(unittest.TestCase):
    def tracker(self, **kwargs):
        defaults = dict(
            xy_size_m=8.0,
            height_m=3.0,
            coverage_complete_ratio=0.9,
            coverage_min_uncovered_cells=3,
            reopen_new_cells=3,
            close_audits=2,
            reopen_audits=2,
            defer_retry_s=10.0,
            max_failures=3,
        )
        defaults.update(kwargs)
        return RegionCoverageTracker(**defaults)

    def test_region_key_is_physical_and_layer_aware(self):
        tracker = self.tracker()
        self.assertEqual(tracker.region_key(1.0, 2.0, 0.5), (0, 0, 0))
        self.assertEqual(tracker.region_key(9.0, 2.0, 0.5), (1, 0, 0))
        self.assertEqual(tracker.region_key(1.0, 2.0, 3.5), (0, 0, 1))

    def test_close_and_reopen_use_hysteresis(self):
        tracker = self.tracker()
        key = (0, 0, 0)
        tracker.update(
            {key: {"frontier_points": 1, "covered_cells": 2, "total_cells": 10}},
            {key},
            0.0,
        )
        self.assertEqual(tracker.status(key), EXPLORING)
        complete = {key: {"covered_cells": 10, "total_cells": 10}}
        tracker.update(complete, {key}, 1.0)
        self.assertEqual(tracker.status(key), EXPLORING)
        tracker.update(complete, {key}, 1.0)
        self.assertEqual(tracker.status(key), EXPLORING)
        tracker.update(complete, {key}, 2.0)
        self.assertEqual(tracker.status(key), COVERED)
        reopened = {
            key: {"frontier_points": 2, "covered_cells": 10, "total_cells": 13}
        }
        tracker.update(reopened, {key}, 3.0)
        self.assertEqual(tracker.status(key), COVERED)
        tracker.update(reopened, {key}, 4.0)
        self.assertEqual(tracker.status(key), EXPLORING)

    def test_covered_region_ignores_frontier_without_graph_growth(self):
        tracker = self.tracker(close_audits=1)
        key = (0, 0, 0)
        tracker.update(
            {key: {"covered_cells": 10, "total_cells": 10}}, {key}, 0.0
        )
        self.assertEqual(tracker.status(key), COVERED)
        for stamp in (1.0, 2.0, 3.0):
            tracker.update(
                {key: {"frontier_points": 4, "covered_cells": 10, "total_cells": 10}},
                {key},
                stamp,
            )
        self.assertEqual(tracker.status(key), COVERED)

    def test_covered_region_reopens_for_stable_feasible_frontier(self):
        tracker = self.tracker(close_audits=1)
        key = (0, 0, 0)
        tracker.update(
            {key: {"covered_cells": 10, "total_cells": 10}}, {key}, 0.0
        )
        evidence = {
            key: {
                "frontier_points": 2,
                "covered_cells": 10,
                "total_cells": 10,
                "selectable": 1,
            }
        }
        tracker.update(evidence, {key}, 1.0)
        self.assertEqual(tracker.status(key), COVERED)
        tracker.update(evidence, {key}, 2.0)
        self.assertEqual(tracker.status(key), EXPLORING)

    def test_unresolved_count_is_scoped_to_reachable_regions(self):
        tracker = self.tracker()
        reachable, orphan = (0, 0, 0), (3, 0, 0)
        evidence = {
            key: {"frontier_points": 1, "covered_cells": 0, "total_cells": 10}
            for key in (reachable, orphan)
        }
        tracker.update(evidence, evidence, 0.0)
        self.assertEqual(tracker.unresolved_count(), 2)
        self.assertEqual(tracker.unresolved_count({reachable}), 1)

    def test_unselectable_regions_are_bounded(self):
        tracker = self.tracker(
            unselectable_close_audits=2,
            blocked_frontier_audits=2,
        )
        residual, blocked = (0, 0, 0), (1, 0, 0)
        tracker.update(
            {
                residual: {"covered_cells": 5, "total_cells": 10},
                blocked: {
                    "frontier_points": 1,
                    "covered_cells": 10,
                    "total_cells": 10,
                },
            },
            {residual, blocked},
            0.0,
        )
        tracker.resolve_unselectable({residual, blocked}, 1.0)
        changed = tracker.resolve_unselectable({residual, blocked}, 2.0)
        self.assertIn((residual, COVERED), changed)
        self.assertIn((blocked, INACCESSIBLE), changed)

    def test_orphaned_unresolved_region_is_bounded(self):
        tracker = self.tracker(orphan_close_audits=2)
        reachable, orphan = (0, 0, 0), (2, 0, 0)
        evidence = {
            key: {"frontier_points": 1, "covered_cells": 2, "total_cells": 10}
            for key in (reachable, orphan)
        }
        tracker.update(evidence, evidence, 0.0)
        tracker.resolve_orphans({reachable}, 1.0)
        changed = tracker.resolve_orphans({reachable}, 2.0)
        self.assertIn((orphan, INACCESSIBLE), changed)
        self.assertEqual(tracker.status(reachable), EXPLORING)

    def test_deferred_region_is_retryable(self):
        tracker = self.tracker()
        key = (0, 0, 0)
        tracker.update(
            {key: {"frontier_points": 2, "covered_cells": 2, "total_cells": 10}},
            {key},
            0.0,
        )
        failures = tracker.defer(key, 1.0)
        self.assertEqual(failures, 1)
        self.assertEqual(tracker.status(key), DEFERRED)
        evidence = {
            key: {"frontier_points": 2, "covered_cells": 2, "total_cells": 10}
        }
        tracker.update(evidence, {key}, 5.0)
        self.assertEqual(tracker.status(key), DEFERRED)
        tracker.update(evidence, {key}, 11.0)
        self.assertEqual(tracker.status(key), EXPLORING)

    def test_route_keeps_preferred_region_and_visits_all(self):
        tracker = self.tracker()
        keys = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        evidence = {
            key: {
                "frontier_points": 2,
                "covered_cells": 2,
                "total_cells": 10,
                "selectable": 1,
            }
            for key in keys
        }
        tracker.update(evidence, keys, 0.0)
        choices = {
            keys[0]: ((1.0, 0.0, 0.0), 1.0),
            keys[1]: ((9.0, 0.0, 0.0), 9.0),
            keys[2]: ((17.0, 0.0, 0.0), 17.0),
        }
        route = tracker.route(choices, (0.0, 0.0, 0.0), preferred_region=keys[1])
        self.assertEqual(route[0], keys[1])
        self.assertEqual(set(route), set(keys))


if __name__ == "__main__":
    unittest.main()
