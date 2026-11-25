#!/usr/bin/env python3

import math
import unittest
from unittest.mock import patch

import numpy as np

from flex_e_core.planner import FlexECore


class FlexECoreTest(unittest.TestCase):
    def planner(self):
        return FlexECore(
            params={
                "visibility_min_voxels": 20,
                "minimum_exploration_s": 0,
                "publish_debug_clouds": False,
                "occupancy_max_range_m": 5.0,
                "occupancy_max_rays": 200,
                "frontier_search_range_m": 8.0,
                "global_frontier_search_range_m": 12.0,
                "frontier_lookahead_m": 2.0,
                "frontier_min_unknown_columns": 1,
                "minimum_waypoint_clearance_m": 0.5,
                "minimum_frontier_waypoint_clearance_m": 0.5,
            }
        )

    def test_waiting_and_terrain_filtering(self):
        planner = self.planner()
        self.assertEqual(planner.step(0.0)["status"], "waiting")
        points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        self.assertEqual(
            planner.integrate_terrain(points, [0.0, 1.0], now_s=0.1), 1
        )
        self.assertEqual(len(planner.cells), 1)
        planner.reset()
        self.assertEqual(len(planner.cells), 0)
        planner.close()

    def test_no_hit_ray_clears_space_without_occupied_endpoint(self):
        planner = self.planner()
        used = planner.integrate_scan(
            np.empty((0, 3)),
            (0.0, 0.0, 0.75),
            now_s=0.1,
            misses_xyz=np.array([[8.0, 0.0, 0.75]]),
        )
        self.assertEqual(used, 1)
        self.assertLess(planner.space.evidence[(2, 0, 2)], 0)
        # The ray clips to the configured five-meter range.
        self.assertLess(planner.space.evidence[(10, 0, 2)], 0)
        planner.close()

    def test_safe_goal_and_controller_failure_feedback(self):
        planner = self.planner()
        xy = np.arange(-4, 5) * 0.5
        terrain = np.array([(x, y, 0.0) for x in xy for y in xy])
        angles = np.linspace(math.pi / 2, 3 * math.pi / 2, 64)
        scan = np.column_stack(
            (3 * np.cos(angles), 3 * np.sin(angles), np.full(64, 0.75))
        )
        self.assertEqual(planner.integrate_terrain(terrain, now_s=0.1), 81)
        planner.set_pose((0.0, 0.0, 0.75), 0.0, 0.1)
        self.assertEqual(
            planner.integrate_scan(scan, (0.0, 0.0, 0.75), now_s=0.2), 64
        )
        decision = planner.step(1.0)
        self.assertEqual(decision["status"], "moving")
        self.assertEqual(decision["kind"], "frontier")
        self.assertIsNotNone(decision["subgoal"])
        self.assertIsNotNone(decision["global_viewpoint"])
        self.assertIsNotNone(decision["anchor"])
        self.assertTrue(math.isfinite(decision["heading_rad"]))
        self.assertTrue(planner._safe_waypoint(planner.active_goal))
        self.assertTrue(planner.report_failure("pointnav_unsolved", now_s=1.1))
        self.assertIsNone(planner.active_goal)
        self.assertIsNone(planner.global_goal)
        self.assertEqual(len(planner.blocked_subgoals), 1)
        planner.close()

    def test_bootstrap_requires_ray_cleared_approach(self):
        planner = self.planner()
        planner.set_pose((0.0, 0.0, 0.75), now_s=0.0)
        planner.integrate_terrain([[2.5, 0.0, 0.0]], now_s=0.0)
        self.assertIsNone(planner._nearest_cell(planner.vehicle))
        self.assertIsNone(planner._bootstrap_waypoint())
        for ix in range(1, 6):
            planner.space.evidence[(ix, 0, 2)] = -3
        planner.space._column_cache.clear()
        self.assertEqual(planner._bootstrap_waypoint(), (5, 0, 0))
        planner.space.evidence[(3, 0, 2)] = 4
        planner.space._column_cache.clear()
        planner.space._clearance_cache.clear()
        self.assertIsNone(planner._bootstrap_waypoint())
        planner.close()

    def test_overlapping_decks_do_not_share_goal_arrival(self):
        planner = self.planner()
        planner.integrate_terrain([[0.0, 0.0, 0.0], [0.0, 0.0, 3.5]], now_s=0.0)
        planner.set_pose((0.0, 0.0, 0.75), now_s=0.0)
        self.assertEqual(planner._nearest_cell(planner.vehicle), (0, 0, 0))
        upper = (0, 0, 14)
        planner._publish_subgoal(upper, upper, 0, 0.0, {}, "test")
        planner.set_pose((0.0, 0.0, 0.75), now_s=1.0)
        self.assertFalse(planner.active_goal_reached_latched)
        self.assertEqual(planner._active_goal_state(1.0), "hold")
        planner.set_pose((0.0, 0.0, 4.25), now_s=2.0)
        self.assertEqual(planner._active_goal_state(2.0), "reached")
        planner.close()

    def test_gap_across_tiles_is_a_target_and_survives_partial_coverage(self):
        planner = FlexECore(params={
            "region_size_m": 2.0,
            "completion_min_reachable_cells": 20,
            "region_coverage_complete_ratio": 0.80,
            "coverage_patch_min_cells": 3,
            "coverage_min_viewpoint_gain": 3,
            "minimum_waypoint_clearance_m": 0.5,
        })
        terrain = [(x * 0.5, y * 0.5, 0.0) for x in range(8) for y in range(4)]
        planner.integrate_terrain(terrain, now_s=0.0)
        for x in range(-15, 24):
            for y in range(-15, 19):
                planner.space.evidence[(x, y, 1)] = -3
        gap = {(3, 1, 0), (4, 1, 0), (5, 1, 0)}
        planner.space.covered_support.update(set(planner.cells) - gap)
        root = (0, 0, 0)
        pose = (0.0, 0.0, 0.75, 0.0)
        first = planner._select_frontier(root, pose, 1.0, resolve_orphans=True)
        self.assertEqual(first[-1], "coverage")
        self.assertEqual(first[-3]["gap_cells"], 3)
        self.assertEqual(set(first[-3]["selected_coverage_keys"]), gap)

        planner.space.covered_support.update({(3, 1, 0), (4, 1, 0)})
        second = planner._select_frontier(root, pose, 2.0, resolve_orphans=True)
        self.assertEqual(second[-1], "coverage")
        self.assertEqual(second[-3]["selected_coverage_keys"], ((5, 1, 0),))
        planner.set_pose(pose[:3], now_s=2.0)
        planner.global_goal = second[1]
        planner.active_region = second[-2]
        planner.global_goal_kind = "coverage"
        planner.global_coverage_keys = second[-3]["selected_coverage_keys"]
        planner._publish_subgoal(second[1], second[1], 0, 2.0, {}, "test")
        previous_revision = planner.subgoal_revision
        planner.space.covered_support.add((5, 1, 0))
        planner.space.coverage_revision += 1
        self.assertIsNone(planner.step(3.0)["subgoal"])
        self.assertIsNone(planner.global_goal)
        self.assertGreater(planner.subgoal_revision, previous_revision)

    def test_failed_observation_marks_only_missed_keys_near_that_viewpoint(self):
        planner = self.planner()
        viewpoint = (6, 6, 0)
        targets = ((8, 6, 0), (9, 6, 0))
        planner.global_goal = viewpoint
        planner.global_goal_kind = "coverage"
        planner.global_coverage_keys = targets
        planner.global_observe_started_s = 1.0
        planner.global_observe_revision = 1
        planner.space.revision = 2
        planner.space.covered_support.add(targets[0])
        self.assertEqual(planner._finish_frontier_observation(2.0), "coverage_region_advanced")
        self.assertEqual(planner._missed_coverage_keys((7, 6, 0)), {targets[1]})
        self.assertEqual(planner._missed_coverage_keys((9, 6, 0)), set())
        self.assertEqual(planner._missed_coverage_keys((6, 6, 14)), set())
        self.assertFalse(planner.space.is_support_key_covered(targets[1]))

    def closure_planner(self):
        planner = FlexECore(params={
            "completion_min_reachable_cells": 20,
            "completion_map_quiet_s": 0,
            "minimum_exploration_s": 0,
            "no_frontier_finish_s": 1.0,
            "global_audit_period_s": 1.0,
            "frontier_search_period_s": 1.0,
            "reachability_recovery_timeout_s": 3.0,
            "coverage_stall_timeout_s": 3.0,
            "visibility_min_voxels": 20,
            "publish_debug_clouds": False,
        })
        for x in range(-16, 48):
            for y in range(-16, 24):
                planner.space.evidence[(x, y, 1)] = -3
        return planner

    def test_detached_audit_preserves_known_regions_and_exits_incomplete(self):
        planner = self.closure_planner()
        patch_cells = [(x / 2.0, y / 2.0, 0.0) for x in range(2) for y in range(2)]
        main_cells = [(x / 2.0, y / 2.0, 0.0) for x in range(20, 26) for y in range(5)]
        planner.integrate_terrain(patch_cells + main_cells, now_s=0.0)
        planner.space.covered_support.update(planner.cells)
        planner.set_pose((0.0, 0.0, 0.75), now_s=0.0)
        planner.last_healthy_component = {(x, y, 0) for x in range(20, 35) for y in range(10)}
        distant_region = planner._region_for_cell((20, 0, 0))
        planner.region_tracker.mark_progress(distant_region)
        for now in (1.0, 2.0, 3.0):
            decision = planner.step(now)
            self.assertIsNone(decision["subgoal"])
            self.assertEqual(decision["diagnostics"]["closure_state"], "INSUFFICIENT_REACHABILITY")
            self.assertEqual(decision["diagnostics"]["reachable_cells"], 4)
        finished = planner.step(4.0)
        self.assertEqual(finished["diagnostics"]["finish_reason"], "reachability_lost")
        self.assertFalse(finished["diagnostics"]["closure_closed"])
        self.assertEqual(planner.region_tracker.unresolved_count({distant_region}), 1)
        self.assertEqual(planner.region_tracker.regions[distant_region].orphan_audits, 0)

    def test_recovery_probes_a_different_safe_attachment(self):
        planner = self.closure_planner()
        terrain = [(x / 2.0, y / 2.0, 0.0) for x in range(8) for y in range(4)]
        planner.integrate_terrain(terrain, now_s=0.0)
        planner.space.covered_support.update(planner.cells)
        planner.set_pose((0.5, 0.5, 0.75), now_s=0.0)
        actual_dijkstra = planner._dijkstra
        wrong_root = (1, 1, 0)

        def attached_component(root, distance, budget):
            if root == wrong_root:
                return {root: None}, {root: 0.0}, 1, False
            return actual_dijkstra(root, distance, budget)

        with patch.object(planner, "_dijkstra", side_effect=attached_component):
            result = planner._select_frontier(
                wrong_root, planner.vehicle, now_s=2.0, resolve_orphans=True
            )
        self.assertGreaterEqual(result[-3]["reachable"], 20)
        self.assertTrue(result[-3]["valid_attachment"])
        self.assertNotEqual(result[-3]["root"], wrong_root)

    def test_healthy_unselectable_gap_exits_without_certifying_closure(self):
        planner = self.closure_planner()
        terrain = [(x / 2.0, y / 2.0, 0.0) for x in range(8) for y in range(4)]
        planner.integrate_terrain(terrain, now_s=0.0)
        planner.set_pose((0.5, 0.5, 0.75), now_s=0.0)
        with patch.object(planner, "_coverage_region_choice", return_value=None):
            first = planner.step(1.0)
            self.assertEqual(first["diagnostics"]["closure_state"], "OPEN")
            for now in (2.0, 3.0):
                result = planner.step(now)
                self.assertIsNone(result["subgoal"])
                self.assertEqual(result["status"], "searching")
            finished = planner.step(4.0)
        self.assertEqual(finished["diagnostics"]["finish_reason"], "coverage_stalled")
        self.assertFalse(finished["diagnostics"]["closure_closed"])

    def test_healthy_closed_audits_can_still_complete(self):
        planner = self.closure_planner()
        terrain = [(x / 2.0, y / 2.0, 0.0) for x in range(8) for y in range(4)]
        planner.integrate_terrain(terrain, now_s=0.0)
        planner.space.covered_support.update(planner.cells)
        planner.set_pose((0.5, 0.5, 0.75), now_s=0.0)
        for now in (1.0, 2.0, 3.0, 4.0, 5.0):
            decision = planner.step(now)
        self.assertEqual(decision["diagnostics"]["finish_reason"], "reachable_space_closed")
        self.assertTrue(decision["diagnostics"]["closure_closed"])

    def test_smaller_audit_does_not_weaken_reference_component(self):
        planner = self.closure_planner()
        terrain = [(x / 2.0, y / 2.0, 0.0) for x in range(8) for y in range(4)]
        planner.integrate_terrain(terrain, now_s=0.0)
        planner.space.covered_support.update(planner.cells)
        planner.set_pose((0.5, 0.5, 0.75), now_s=0.0)
        planner.last_healthy_component = {
            (x, y, 0) for x in range(20) for y in range(6)
        }
        planner._select_frontier((1, 1, 0), planner.vehicle, now_s=2.0, resolve_orphans=True)
        self.assertEqual(planner._minimum_audit_reachable(), 30)
        self.assertFalse(planner._valid_audit_attachment(set(sorted(planner.cells)[:29])))


if __name__ == "__main__":
    unittest.main()
