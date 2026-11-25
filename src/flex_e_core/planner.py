"""ROS-independent FLEX-E coverage and safe viewpoint planner.

Input points and poses use a common x/y horizontal, z-up metric frame.
Terrain input must already represent traversable support surfaces.
"""
import csv
import heapq
import logging
import math
import os
import random
import threading
import time
from collections import defaultdict, deque

import numpy as np

from .region_coverage import (
    COVERED, DEFERRED, EXPLORING, INACCESSIBLE, RegionCoverageTracker,
)
from .space_evaluator import FREE, OCCUPIED, ExplorationSpaceEvaluator


class FlexECore(object):
    """Layer-aware Dijkstra frontier selection over observed terrain support."""

    def __init__(self, params=None, logger=None, debug_callback=None, param_source=None):
        self._params = dict(params or {})
        self._param_source = param_source
        self._logger = logger or logging.getLogger(__name__)
        self._debug_callback = debug_callback
        self._log_throttles = {}
        self._last_row = {}
        self._last_runtime_s = 0.0
        self._closure_closed = False
        self._finish_reason = ""
        def param(name, default):
            if name in self._params:
                return self._params[name]
            if self._param_source is not None:
                return self._param_source(name, default)
            return default
        self.terrain_topic = param("terrain_topic", "/terrain_map_ext")
        self.registered_scan_topic = param("registered_scan_topic", "/registered_scan")
        self.scan_odom_topic = param("scan_state_estimation_topic", "/state_estimation_at_scan")
        self.odom_topic = param("state_estimation_topic", "/state_estimation")
        self.waypoint_topic = param("waypoint_topic", "/way_point")
        self.runtime_topic = param("runtime_topic", "/runtime")
        self.finish_topic = param("exploration_finish_topic", "/exploration_finish")
        self.stop_topic = param("stop_exploring_topic", "/stop_exploring")
        self.frame_id = param("frame_id", "map")
        self.grid_m = max(0.15, float(param("grid_resolution_m", 0.5)))
        self.z_grid_m = max(0.05, float(param("z_resolution_m", 0.25)))
        self.max_slope = math.radians(float(param("max_slope_deg", 20.0)))
        self.max_step_m = max(0.0, float(param("max_step_m", 0.25)))
        self.safe_intensity = max(0.0, float(param("safe_intensity_m", 0.15)))
        self.vehicle_height_m = float(param("vehicle_height_m", 0.75))
        self.max_cells = max(1000, int(param("max_graph_cells", 160000)))
        self.max_expansions = max(100, int(param("max_search_nodes", 5000)))
        self.search_range_m = max(2.0, float(param("frontier_search_range_m", 25.0)))
        self.global_search_range_m = max(
            self.search_range_m, float(param("global_frontier_search_range_m", 120.0))
        )
        self.global_max_expansions = max(
            self.max_expansions, int(param("global_max_search_nodes", 30000))
        )
        self.period_s = max(0.1, float(param("planner_period_s", 1.0)))
        self.goal_min_distance_m = max(0.0, float(param("minimum_goal_distance_m", 1.0)))
        self.cleanup_goal_distance_m = max(
            0.0,
            min(self.goal_min_distance_m, float(param("cleanup_goal_distance_m", 0.75))),
        )
        self.visited_radius_m = max(0.0, float(param("visited_radius_m", 0.75)))
        self.visited_vertical_tolerance_m = max(0.0, float(param("visited_vertical_tolerance_m", 0.75)))
        self.visited_penalty = max(0.0, float(param("visited_penalty", 1.0)))
        self.visit_sample_distance_m = max(0.1, float(param("visit_sample_distance_m", 0.75)))
        self.execution_horizon_m = max(0.25, float(param("execution_horizon_m", 3.0)))
        self.frontier_standoff_m = max(0.0, float(param("frontier_standoff_m", 2.0)))
        self.max_frontier_gain = max(1, int(param("max_frontier_gain", 2)))
        self.min_frontier_support = max(1, int(param("min_frontier_support_neighbors", 3)))
        self.completed_reopen_unknown_columns = max(
            1, int(param("completed_reopen_unknown_columns", 6))
        )
        self.frontier_cluster_link_m = max(
            self.grid_m, float(param("frontier_cluster_link_m", 1.0))
        )
        self.frontier_cluster_vertical_m = max(
            self.z_grid_m,
            float(param("frontier_cluster_vertical_tolerance_m", 0.75)),
        )
        self.frontier_cluster_gain_reward = max(
            0.0, float(param("frontier_cluster_gain_reward", 0.20))
        )
        self.frontier_observation_wait_s = max(
            self.period_s, float(param("frontier_observation_wait_s", 2.0))
        )
        self.frontier_progress_radius_m = max(
            self.grid_m, float(param("frontier_progress_radius_m", 3.0))
        )
        self.frontier_min_growth_cells = max(
            1, int(param("frontier_min_growth_cells", 20))
        )
        self.global_goal_reached_m = max(0.1, float(param("global_goal_reached_m", 1.25)))
        self.goal_z_tolerance_m = max(self.z_grid_m, float(param("goal_z_tolerance_m", 0.55)))
        self.portal_probe_range_m = max(0.0, float(param("portal_probe_range_m", 2.25)))
        self.frontier_gain_rate_selection = bool(param("frontier_gain_rate_selection", False))
        self.blocked_subgoal_radius_m = max(0.1, float(param("blocked_subgoal_radius_m", 2.0)))
        self.blocked_subgoal_memory_s = max(1.0, float(param("blocked_subgoal_memory_s", 300.0)))
        self.distance_reward = float(param("frontier_distance_reward", 0.08))
        self.heading_reward = float(param("frontier_heading_reward", 1.5))
        self.goal_hold_s = max(self.period_s, float(param("active_goal_min_hold_s", 12.0)))
        self.goal_reached_m = max(0.05, float(param("active_goal_reached_m", 0.75)))
        self.goal_stall_s = max(self.goal_hold_s, float(param("active_goal_stall_s", 20.0)))
        self.goal_timeout_s = max(self.goal_stall_s, float(param("active_goal_timeout_s", 20.0)))
        self.progress_epsilon_m = max(0.01, float(param("progress_epsilon_m", 0.25)))
        self.map_growth_min_cells = max(1, int(param("map_growth_min_cells", 5)))
        self.no_frontier_finish_s = max(1.0, float(param("no_frontier_finish_s", 20.0)))
        self.frontier_search_period_s = max(
            self.period_s, float(param("frontier_search_period_s", 5.0))
        )
        self.global_audit_period_s = max(
            self.frontier_search_period_s,
            float(param("global_audit_period_s", 10.0)),
        )
        self.completion_map_quiet_s = max(
            0.0, float(param("completion_map_quiet_s", 20.0))
        )
        self.completion_stable_audits = max(
            1, int(param("completion_stable_audits", 3))
        )
        self.completion_search_range_m = max(
            self.global_search_range_m,
            float(param("completion_search_range_m", 200.0)),
        )
        self.completion_max_expansions = max(
            self.global_max_expansions,
            int(param("completion_max_search_nodes", self.max_cells)),
        )
        self.completion_min_reachable_cells = max(
            20, int(param("completion_min_reachable_cells", 100))
        )
        self.recovery_timeout_s = max(
            2.0 * self.frontier_search_period_s, float(param("reachability_recovery_timeout_s", 45.0))
        )
        self.recovery_attempt_limit = max(1, int(param("reachability_recovery_attempts", 3)))
        self.coverage_stall_timeout_s = max(
            self.global_audit_period_s, float(param("coverage_stall_timeout_s", 90.0))
        )
        self.minimum_exploration_s = max(0.0, float(param("minimum_exploration_s", 120.0)))
        self.max_exploration_s = max(0.0, float(param("max_exploration_s", 3600.0)))
        self.shutdown_on_finish = bool(param("shutdown_on_finish", False))
        self.finish_shutdown_delay_s = max(0.1, float(param("finish_shutdown_delay_s", 2.0)))
        self.trace_idle_period_s = max(self.period_s, float(param("trace_idle_period_s", 2.0)))
        self.rng = random.Random(int(param("seed", 0)))
        self.trace_path = str(param("decision_trace_path", "")).strip()

        # TARE-style coarse physical subspaces. These states are independent
        # of transient frontier cluster IDs and use hysteresis to prevent a
        # weak observation from repeatedly opening and closing a region.
        self.region_size_m = max(2.0, float(param("region_size_m", 8.0)))
        self.region_height_m = max(0.5, float(param("region_height_m", 3.0)))
        self.coverage_max_points_per_region = max(
            32, int(param("coverage_max_points_per_region", 256))
        )
        self.coverage_candidate_limit = max(
            4, int(param("coverage_candidate_limit", 32))
        )
        self.coverage_min_viewpoint_gain = max(
            1, int(param("coverage_min_viewpoint_gain", 6))
        )
        self.coverage_min_observation_gain = max(
            1, int(param("coverage_min_observation_gain", 4))
        )
        self.coverage_patch_min_cells = max(
            0, int(param("coverage_patch_min_cells", 0))
        )
        self.coverage_prediction_min_known_fraction = min(
            1.0,
            max(
                0.0,
                float(param("coverage_prediction_min_known_fraction", 0.35)),
            ),
        )
        self.coverage_gain_reward = max(
            0.0, float(param("coverage_gain_reward", 2.0))
        )
        self.clearance_reward = max(
            0.0, float(param("waypoint_clearance_reward", 1.25))
        )
        self.minimum_waypoint_clearance_m = max(
            self.grid_m,
            float(param("minimum_waypoint_clearance_m", 0.75)),
        )
        self.hard_traversal_clearance_m = max(
            0.0, float(param("hard_traversal_clearance_m", 0.45))
        )
        self.minimum_frontier_waypoint_clearance_m = max(
            self.hard_traversal_clearance_m,
            float(param("minimum_frontier_waypoint_clearance_m", 0.55)),
        )
        self.portal_exit_distance_m = max(
            self.grid_m, float(param("portal_exit_distance_m", 1.5))
        )
        self.portal_exit_search_m = max(
            self.portal_exit_distance_m,
            float(param("portal_exit_search_m", 10.0)),
        )

        self.visibility_update_period_s = max(
            0.1, float(param("visibility_update_period_s", 0.5))
        )
        self.visibility_min_voxels = max(
            20, int(param("visibility_min_voxels", 100))
        )
        self.occupancy_growth_min_voxels = max(
            1, int(param("occupancy_growth_min_voxels", 20))
        )
        self.completed_region_radius_m = max(
            self.grid_m, float(param("completed_region_radius_m", 3.0))
        )
        self.completed_region_progress_radius_m = max(
            self.grid_m,
            min(
                self.completed_region_radius_m,
                float(param("completed_region_progress_radius_m", 1.5)),
            ),
        )
        self.space = ExplorationSpaceEvaluator(
            resolution_m=param("occupancy_resolution_m", 0.5),
            ray_step_m=param("occupancy_ray_step_m", 0.5),
            min_range_m=param("occupancy_min_range_m", 0.75),
            max_range_m=param("occupancy_max_range_m", 15.0),
            max_rays=param("occupancy_max_rays", 900),
            occupied_threshold=param("occupancy_occupied_threshold", 2),
            free_threshold=param("occupancy_free_threshold", -1),
            obstacle_inflation_m=param("occupancy_obstacle_inflation_m", 0.60),
            min_clearance_m=param("occupancy_min_clearance_m", 0.30),
            max_clearance_m=param("occupancy_max_clearance_m", 1.60),
            frontier_lookahead_m=param("frontier_lookahead_m", 4.0),
            frontier_lateral_m=param("frontier_lateral_m", 1.0),
            frontier_min_unknown_columns=param("frontier_min_unknown_columns", 3),
            max_voxels=param("occupancy_max_voxels", 1200000),
            completed_resolution_m=param("completed_region_resolution_m", 1.0),
            completed_vertical_tolerance_m=param(
                "completed_region_vertical_tolerance_m", 0.75
            ),
            support_coverage_resolution_m=param(
                "support_coverage_resolution_m", 0.5
            ),
            support_coverage_range_m=param("support_coverage_range_m", 12.0),
            support_coverage_max_cells=param(
                "support_coverage_max_cells", 200000
            ),
            support_coverage_min_known_fraction=param(
                "support_coverage_min_known_fraction", 0.65
            ),
            clearance_search_m=param("clearance_search_m", 3.0),
        )
        self.region_tracker = RegionCoverageTracker(
            xy_size_m=self.region_size_m,
            height_m=self.region_height_m,
            height_origin_m=param("region_height_origin_m", 0.0),
            open_frontier_points=param("region_open_frontier_points", 1),
            close_frontier_points=param("region_close_frontier_points", 1),
            reopen_frontier_points=param("region_reopen_frontier_points", 2),
            coverage_complete_ratio=param("region_coverage_complete_ratio", 0.92),
            coverage_min_uncovered_cells=param(
                "region_coverage_min_uncovered_cells", 6
            ),
            reopen_new_cells=param("region_reopen_new_cells", 12),
            unselectable_close_audits=param(
                "region_unselectable_close_audits", 2
            ),
            blocked_frontier_audits=param(
                "region_blocked_frontier_audits", 3
            ),
            orphan_close_audits=param("region_orphan_close_audits", 3),
            close_audits=param("region_close_audits", 3),
            reopen_audits=param("region_reopen_audits", 2),
            defer_retry_s=param("region_defer_retry_s", 30.0),
            max_failures=param("region_max_failures", 3),
        )
        self.publish_debug_clouds = bool(param("publish_debug_clouds", True))

        # key -> [mean support elevation, bounded observation count, last stamp]
        self.cells = {}
        self.cell_first_seen_s = {}
        self.levels = defaultdict(set)  # (x-cell, y-cell) -> layer keys
        self.graph_revision = 0
        self.last_graph_growth_s = None
        self.last_space_growth_s = None
        self.last_visibility_update_s = None
        self.vehicle = None
        self.vehicle_yaw = 0.0
        self.visited_index = defaultdict(set)
        radius_cells = int(math.ceil(self.visited_radius_m / self.grid_m))
        self.visit_offsets = [
            (dx, dy)
            for dx in range(-radius_cells, radius_cells + 1)
            for dy in range(-radius_cells, radius_cells + 1)
            if (dx * self.grid_m) ** 2 + (dy * self.grid_m) ** 2 <= self.visited_radius_m ** 2
        ]
        self.last_visit = None
        self.recent_roots = deque(maxlen=40)
        self.recovery_until_s = {}
        self.last_healthy_component = set()
        self.recovery_started_s = None
        self.recovery_attempts = 0
        self.coverage_stall_started_s = None
        self.coverage_stall_revision = None
        self.portal_retry_after_s = {}
        self.missed_coverage_by_viewpoint = {}
        self.pending_gap_keys = set()
        self.active_goal = None
        self.subgoal_revision = 0
        self.active_goal_reached_latched = False
        self.active_started_s = None
        self.active_last_progress_s = None
        self.active_best_distance_m = None
        self.global_goal = None
        self.frontier_anchor = None
        self.frontier_anchor_directions = ()
        self.frontier_unknown_before = 0
        self.global_goal_gain = 0
        self.global_goal_kind = ""
        self.global_coverage_before = 0
        self.global_coverage_revision = 0
        self.global_coverage_keys = ()
        self.global_region_support_keys = ()
        self.active_region = None
        self.preferred_region = None
        self.planned_region_route = []
        self.global_goal_search_range_m = self.search_range_m
        self.global_goal_max_expansions = self.max_expansions
        self.blocked_subgoals = []
        self.global_goal_selected_s = None
        self.global_observe_started_s = None
        self.global_observe_revision = 0
        self.last_goal = None
        self.exploration_started_s = None
        self.no_frontier_started_s = None
        self.completion_audit_count = 0
        self.next_frontier_search_s = 0.0
        self.last_idle_trace_s = -float("inf")
        self.finished = False
        self.shutdown_timer = None
        self.trace_file = None
        self.trace_writer = None
        self.state_lock = threading.RLock()
        self._open_trace()

        self._log_info(
            "FLEX-E ready: grid=%.2f/%.2f m", self.grid_m, self.z_grid_m
        )

    def _open_trace(self):
        if not self.trace_path:
            return
        fields = (
            "sim_time_s",
            "decision_ms",
            "cell_count",
            "occupancy_voxels",
            "completed_region_cells",
            "expanded",
            "frontiers",
            "frontier_regions",
            "selectable_frontiers",
            "selectable_coverage",
            "uncovered_support_cells",
            "connected_gap_cells",
            "unresolved_regions",
            "coverage_ratio",
            "exploring_regions",
            "covered_regions",
            "deferred_regions",
            "inaccessible_regions",
            "orphaned_regions",
            "global_region_route",
            "selected_region",
            "target_kind",
            "reachable_cells",
            "audit_root",
            "minimum_audit_reachable",
            "recovery_attempts",
            "search_truncated",
            "closure_state",
            "information_gain",
            "target_x",
            "target_y",
            "target_z",
            "reason",
        )

        # Never silently append new rows through an old CSV schema.  The old
        # behavior reused the existing header and discarded newly added
        # diagnostics through extrasaction="ignore", which made a new run look
        # as if it still used the old planner.  Preserve the old file and pick
        # a deterministic schema-suffixed sibling instead.
        trace_path = self.trace_path
        suffix = 2
        while os.path.isfile(trace_path) and os.path.getsize(trace_path) > 0:
            with open(trace_path, "r", newline="", encoding="utf-8") as existing:
                existing_header = existing.readline().strip()
            try:
                existing_fields = tuple(next(csv.reader([existing_header])))
            except (csv.Error, StopIteration):
                existing_fields = ()
            if existing_fields == fields:
                break
            root, extension = os.path.splitext(self.trace_path)
            trace_path = "%s_schema%d%s" % (root, suffix, extension)
            suffix += 1
        if trace_path != self.trace_path:
            self._log_warning(
                "FLEX-E trace schema changed; preserving %s and writing %s",
                self.trace_path,
                trace_path,
            )
            self.trace_path = trace_path

        parent = os.path.dirname(os.path.abspath(self.trace_path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        self.trace_file = open(self.trace_path, "a+", newline="", encoding="utf-8")
        self.trace_file.seek(0)
        existing_header = self.trace_file.readline().strip()
        self.trace_file.seek(0, os.SEEK_END)
        self.trace_writer = csv.DictWriter(
            self.trace_file,
            fieldnames=fields,
            extrasaction="ignore",
        )
        if self.trace_file.tell() == 0:
            self.trace_writer.writeheader()
            self.trace_file.flush()

    def _close_trace(self):
        if self.trace_file:
            self.trace_file.close()
            self.trace_file = None

    def _trace(self, row):
        if self.trace_writer:
            self.trace_writer.writerow(row)
            self.trace_file.flush()

    def set_pose(self, xyz, yaw_rad=0.0, now_s=None):
        """Update pose in the x/y horizontal, z-up map frame."""
        point = np.asarray(xyz, dtype=np.float64).reshape(3)
        if not np.isfinite(point).all():
            raise ValueError("pose must contain finite x/y/z")
        stamp = self._clock(now_s)
        with self.state_lock:
            previous = self.vehicle
            self.vehicle = (float(point[0]), float(point[1]), float(point[2]), stamp)
            self.vehicle_yaw = float(yaw_rad)

            # Preserve the original high-rate waypoint crossing latch.
            if self.active_goal in self.cells:
                goal_x = self.active_goal[0] * self.grid_m
                goal_y = self.active_goal[1] * self.grid_m
                goal_z = self.cells[self.active_goal][0] + self.vehicle_height_m
                distance = math.hypot(goal_x - point[0], goal_y - point[1])
                same_height = abs(goal_z - point[2]) <= self.goal_z_tolerance_m
                arrival_radius = self._active_arrival_radius()
                crossed = same_height and distance <= arrival_radius
                if (not crossed and same_height and previous is not None
                        and abs(goal_z - previous[2]) <= self.goal_z_tolerance_m):
                    crossed = self._point_segment_distance(
                        goal_x, goal_y, previous[0], previous[1], point[0], point[1]
                    ) <= arrival_radius
                if crossed:
                    self.active_goal_reached_latched = True
            if (
                self.last_visit is None
                or math.hypot(point[0] - self.last_visit[0], point[1] - self.last_visit[1])
                >= self.visit_sample_distance_m
            ):
                ix = int(math.floor(point[0] / self.grid_m + 0.5))
                iy = int(math.floor(point[1] / self.grid_m + 0.5))
                ground_z = point[2] - self.vehicle_height_m
                iz = int(math.floor(ground_z / self.z_grid_m + 0.5))
                self.visited_index[(ix, iy)].add(iz)
                self.last_visit = tuple(float(value) for value in point)

    @staticmethod
    def _point_segment_distance(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq <= 1.0e-12:
            return math.hypot(px - ax, py - ay)
        ratio = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
        return math.hypot(px - (ax + ratio * dx), py - (ay + ratio * dy))

    def integrate_terrain(self, xyz, intensity=None, now_s=None):
        """Add observed traversable support points, not arbitrary depth returns."""
        points = np.asarray(xyz, dtype=np.float64)
        if points.size == 0:
            return 0
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("terrain xyz must have shape (N, 3)")
        if intensity is None:
            intensity = np.zeros(len(points), dtype=np.float64)
        else:
            intensity = np.asarray(intensity, dtype=np.float64).reshape(-1)
            if len(intensity) != len(points):
                raise ValueError("terrain intensity must match xyz length")
        with self.state_lock:
            return self._integrate_terrain_locked(points, intensity, self._clock(now_s))

    def _integrate_terrain_locked(self, xyz, intensity, stamp):
        valid = np.isfinite(xyz).all(axis=1) & np.isfinite(intensity) & (intensity <= self.safe_intensity)
        xyz = xyz[valid]
        if not len(xyz):
            return 0
        ix = np.floor(xyz[:, 0] / self.grid_m + 0.5).astype(np.int64)
        iy = np.floor(xyz[:, 1] / self.grid_m + 0.5).astype(np.int64)
        iz = np.floor(xyz[:, 2] / self.z_grid_m + 0.5).astype(np.int64)
        keys = np.column_stack((ix, iy, iz))
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        z_sums = np.bincount(inverse, weights=xyz[:, 2])
        new_count = 0
        for row, count, z_sum in zip(unique, counts, z_sums):
            key = (int(row[0]), int(row[1]), int(row[2]))
            elevation = float(z_sum / count)
            previous = self.cells.get(key)
            if previous is None:
                if len(self.cells) >= self.max_cells:
                    self._trim_old_cells()
                self.cells[key] = [elevation, min(20, int(count)), stamp]
                self.cell_first_seen_s[key] = stamp
                self.levels[key[:2]].add(key)
                new_count += 1
            else:
                old_weight = min(20, int(previous[1]))
                new_weight = min(20, int(count))
                previous[0] = (previous[0] * old_weight + elevation * new_weight) / (old_weight + new_weight)
                previous[1] = min(20, old_weight + new_weight)
                previous[2] = stamp
        if new_count:
            self.graph_revision += new_count
            if new_count >= self.map_growth_min_cells:
                self.last_graph_growth_s = stamp
                self.completion_audit_count = 0
                self.no_frontier_started_s = None
        return new_count

    def integrate_scan(self, xyz, origin_xyz, now_s=None, misses_xyz=None):
        """Integrate returns and optional no-hit rays, then update coverage."""
        points = np.asarray(xyz, dtype=np.float64)
        if points.size == 0:
            points = np.empty((0, 3), dtype=np.float64)
        elif points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("scan xyz must have shape (N, 3)")
        if misses_xyz is None:
            misses = None
        else:
            misses = np.asarray(misses_xyz, dtype=np.float64)
            if misses.size == 0:
                misses = np.empty((0, 3), dtype=np.float64)
            elif misses.ndim != 2 or misses.shape[1] != 3:
                raise ValueError("no-hit xyz must have shape (N, 3)")
        if not len(points) and (misses is None or not len(misses)):
            return 0
        origin = np.asarray(origin_xyz, dtype=np.float64).reshape(3)
        if not np.isfinite(origin).all():
            raise ValueError("scan origin must contain finite x/y/z")
        stamp = self._clock(now_s)
        with self.state_lock:
            if self.last_visibility_update_s is not None:
                delta_s = stamp - self.last_visibility_update_s
                if 0.0 <= delta_s < self.visibility_update_period_s:
                    return 0
            voxel_count_before = len(self.space.evidence)
            integrated = self.space.integrate_scan(
                origin, points, misses_xyz=misses
            )
            self._observe_support_from_scan(origin)
            if (
                len(self.space.evidence) - voxel_count_before
                >= self.occupancy_growth_min_voxels
            ):
                self.last_space_growth_s = stamp
                self.completion_audit_count = 0
                self.no_frontier_started_s = None
            self.last_visibility_update_s = stamp
            return integrated

    def _observe_support_from_scan(self, origin):
        """Mark traversable support visible from this registered scan pose."""

        radius_cells = int(
            math.ceil(self.space.support_coverage_range_m / self.grid_m)
        )
        center_x = int(math.floor(float(origin[0]) / self.grid_m + 0.5))
        center_y = int(math.floor(float(origin[1]) / self.grid_m + 0.5))
        support = []
        for ix in range(center_x - radius_cells, center_x + radius_cells + 1):
            for iy in range(center_y - radius_cells, center_y + radius_cells + 1):
                if math.hypot(ix - center_x, iy - center_y) * self.grid_m > self.space.support_coverage_range_m:
                    continue
                for key in self.levels.get((ix, iy), ()):
                    support.append((ix * self.grid_m, iy * self.grid_m, self.cells[key][0]))
        self.space.observe_support_cells(origin, support, self.vehicle_height_m)

    def _trim_old_cells(self):
        for key in sorted(self.cells, key=lambda item: self.cells[item][2])[:max(1, self.max_cells // 20)]:
            self.cells.pop(key, None)
            self.cell_first_seen_s.pop(key, None)
            self.missed_coverage_by_viewpoint.pop(key, None)
            levels = self.levels.get(key[:2])
            if levels:
                levels.discard(key)
                if not levels:
                    self.levels.pop(key[:2], None)

    def _attachment_candidates(self, vehicle, radius_cells=2):
        ix, iy = (int(math.floor(value / self.grid_m + 0.5)) for value in vehicle[:2])
        ground_z = vehicle[2] - self.vehicle_height_m
        candidates = []
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                for key in self.levels.get((ix + dx, iy + dy), ()):
                    dz = abs(self.cells[key][0] - ground_z)
                    if dz <= self.goal_z_tolerance_m:
                        x, y = key[0] * self.grid_m, key[1] * self.grid_m
                        distance = math.hypot(x - vehicle[0], y - vehicle[1])
                        candidates.append((distance ** 2 + dz ** 2, key))
        return [key for _, key in sorted(candidates)]

    def _nearest_cell(self, vehicle):
        nearest, best_size, tested = None, 0, 0
        for key in self._attachment_candidates(vehicle):
            if not self._pose_to_support_clear(vehicle, key):
                continue
            _, connected, _, _ = self._dijkstra(key, 3.0, 16)
            if len(connected) > best_size:
                nearest, best_size = key, len(connected)
            tested += 1
            if best_size >= 8 or tested >= 6:
                break
        return nearest

    def _minimum_audit_reachable(self):
        # Reject a sudden collapse of a previously audited component even if
        # the remaining fragment is large enough to pass the startup minimum.
        return max(self.completion_min_reachable_cells, len(self.last_healthy_component) // 4)

    def _valid_audit_attachment(self, connected):
        return len(connected) >= self._minimum_audit_reachable() and (
            not self.last_healthy_component
            or not self.last_healthy_component.isdisjoint(connected)
        )

    def _connected_recovery_anchor(self, vehicle, candidates, rejected, now_s, move=False):
        """Check a few distinct, safely attachable components under the current map."""
        checked = set(rejected)
        probes = 0
        required = self._minimum_audit_reachable()
        for key in candidates:
            if key in checked or key not in self.cells:
                continue
            if move and now_s < self.recovery_until_s.get(key, 0.0):
                continue
            x, y, z = self._key_to_pose(key)
            distance = math.hypot(x - vehicle[0], y - vehicle[1])
            if abs(z - vehicle[2]) > self.goal_z_tolerance_m:
                continue
            if move:
                if not (self.goal_reached_m + 0.25 <= distance <= 5.0):
                    continue
                if not self._safe_waypoint(key):
                    continue
            elif distance > max(1.5, 2.0 * self.grid_m):
                continue
            if not self._pose_to_support_clear(vehicle, key, check_support_edges=True):
                continue
            _, connected, _, _ = self._dijkstra(key, float("inf"), required)
            if self._valid_audit_attachment(connected):
                return key
            checked.update(connected)
            probes += 1
            if probes >= 4:
                break
        return None

    def _pose_to_support_clear(self, vehicle, target, check_support_edges=False):
        """Check the known support and footprint along a short attachment."""
        target_x, target_y = target[0] * self.grid_m, target[1] * self.grid_m
        target_z = self.cells[target][0]
        ground_z = vehicle[2] - self.vehicle_height_m
        distance = math.hypot(target_x - vehicle[0], target_y - vehicle[1])
        slope_guard = self.max_step_m + math.tan(self.max_slope) * distance
        if abs(target_z - ground_z) > max(self.goal_z_tolerance_m, slope_guard):
            return False
        steps = max(1, int(math.ceil(2.0 * distance / self.grid_m)))
        previous_support = None
        for step in range(1, steps + 1):
            progress = step / steps
            x = vehicle[0] + progress * (target_x - vehicle[0])
            y = vehicle[1] + progress * (target_y - vehicle[1])
            z = ground_z + progress * (target_z - ground_z)
            ix = int(math.floor(x / self.grid_m + 0.5))
            iy = int(math.floor(y / self.grid_m + 0.5))
            supports = self.levels.get((ix, iy), ())
            support = min(
                supports,
                key=lambda key: (abs(self.cells[key][0] - z), key),
                default=None,
            )
            if support is None or abs(self.cells[support][0] - z) > self.goal_z_tolerance_m:
                return False
            if (
                check_support_edges
                and previous_support is not None
                and previous_support != support
                and support not in self._neighbors(previous_support)
            ):
                return False
            previous_support = support
            if self.space.column_state(
                x, y, z, self.hard_traversal_clearance_m
            ) == OCCUPIED:
                return False
        return True

    def _recover_reachability(self, rejected, now_s, row):
        self.completion_audit_count = 0
        self.no_frontier_started_s = None
        self.coverage_stall_started_s = None
        self.next_frontier_search_s = now_s + self.frontier_search_period_s
        if self.recovery_started_s is None:
            self.recovery_started_s = now_s
            self._log_warning(
                "FLEX-E recovering graph attachment: root=%s reachable=%d required=%d",
                row["audit_root"], len(rejected), self._minimum_audit_reachable(),
            )
        row["recovery_attempts"] = self.recovery_attempts
        if now_s - self.recovery_started_s >= self.recovery_timeout_s:
            self._finish_exploration("reachability_lost", now_s, row)
            return
        row["reason"] = "closure_audit_insufficient_reachability"
        if self.recovery_attempts >= self.recovery_attempt_limit:
            return
        key = self._connected_recovery_anchor(
            self.vehicle, reversed(self.recent_roots), rejected, now_s, move=True
        )
        if key is not None:
            self.recovery_attempts += 1
            row["recovery_attempts"] = self.recovery_attempts
            self.recovery_until_s[key] = now_s + self.blocked_subgoal_memory_s
            self.global_goal_kind = "recovery"
            self.next_frontier_search_s = 0.0
            self._publish_subgoal(key, key, 0, now_s, row, "recover_connected_support")

    def _bootstrap_waypoint(self):
        """Enter nearby measured ground across the initial LiDAR blind zone."""
        if self.recent_roots:
            return None
        vehicle = self.vehicle
        ix, iy = (int(math.floor(value / self.grid_m + 0.5)) for value in vehicle[:2])
        limit_m = 3.5
        radius = int(math.ceil(limit_m / self.grid_m))
        candidates = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for key in self.levels.get((ix + dx, iy + dy), ()):
                    point = self._key_to_pose(key)
                    distance = math.hypot(point[0] - vehicle[0], point[1] - vehicle[1])
                    if 0.6 < distance <= limit_m and abs(point[2] - vehicle[2]) <= self.goal_z_tolerance_m:
                        candidates.append((distance, key, point))
        for distance, key, point in sorted(candidates):
            if not self._safe_waypoint(key, self.minimum_frontier_waypoint_clearance_m):
                continue
            steps = int(math.ceil(2.0 * distance / self.grid_m))
            for step in range(1, steps + 1):
                fraction = step / steps
                x, y, z = (
                    vehicle[i] + fraction * (point[i] - vehicle[i]) for i in range(3)
                )
                state = self.space.column_state(
                    x, y, z - self.vehicle_height_m, self.hard_traversal_clearance_m
                )
                if state == OCCUPIED or (
                    state != FREE and fraction * distance >= self.space.min_range_m
                ):
                    break
            else:
                return key
        return None

    def _edge_ok(self, first, second):
        horizontal = self.grid_m * math.hypot(first[0] - second[0], first[1] - second[1])
        if horizontal <= 0.0:
            return False
        dz = abs(self.cells[first][0] - self.cells[second][0])
        return dz <= self.max_step_m + math.tan(self.max_slope) * horizontal + 1.0e-6

    def _neighbors(self, key):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if not dx and not dy:
                    continue
                for candidate in self.levels.get((key[0] + dx, key[1] + dy), ()):
                    if not self._edge_ok(key, candidate):
                        continue
                    x, y = candidate[0] * self.grid_m, candidate[1] * self.grid_m
                    if self.space.column_state(
                        x,
                        y,
                        self.cells[candidate][0],
                        self.hard_traversal_clearance_m,
                    ) == OCCUPIED:
                        continue
                    # Do not cut diagonally between two inflated obstacle
                    # columns. This makes graph reachability respect the
                    # vehicle footprint in narrow passages.
                    if dx and dy:
                        side_blocked = 0
                        for side_xy in ((key[0] + dx, key[1]), (key[0], key[1] + dy)):
                            side_levels = self.levels.get(side_xy, ())
                            if not any(
                                self._edge_ok(key, side)
                                and self.space.column_state(
                                    side[0] * self.grid_m,
                                    side[1] * self.grid_m,
                                    self.cells[side][0],
                                    self.hard_traversal_clearance_m,
                                )
                                != OCCUPIED
                                for side in side_levels
                            ):
                                side_blocked += 1
                        if side_blocked == 2:
                            continue
                    yield candidate

    def _frontier_evidence(self, key, include_completed=False):
        """Return visible unknown directions, information gain and risk."""

        directions = []
        unknown_columns = 0
        occupied_columns = 0
        elevation = self.cells[key][0]
        tolerance = self.max_step_m + self.grid_m * math.tan(self.max_slope)
        for dx, dy in (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ):
            candidates = self.levels.get((key[0] + dx, key[1] + dy), ())
            if any(abs(self.cells[item][0] - elevation) <= tolerance for item in candidates):
                continue
            x, y = key[0] * self.grid_m, key[1] * self.grid_m
            completed = self.space.is_completed(
                x, y, elevation, dx, dy
            )
            evidence = self.space.direction_evidence(
                x, y, elevation, dx, dy, self.grid_m
            )
            if not evidence.is_frontier:
                continue
            if (
                completed
                and not include_completed
                and evidence.unknown_columns < self.completed_reopen_unknown_columns
            ):
                continue
            directions.append((dx, dy))
            unknown_columns += evidence.unknown_columns
            occupied_columns += evidence.occupied_columns
        return tuple(directions), unknown_columns, occupied_columns

    def _frontier_gain(self, key):
        directions, _unknown_columns, _occupied_columns = self._frontier_evidence(key)
        return len(directions)

    def _portal_frontier_evidence(self, key, now_s, ignore_cooldown=False):
        if self.portal_probe_range_m <= 0.0:
            return (), 0, 0
        if not ignore_cooldown and now_s < self.portal_retry_after_s.get(key, 0.0):
            return (), 0, 0
        x, y = key[0] * self.grid_m, key[1] * self.grid_m
        elevation = self.cells[key][0]
        directions = []
        information = 0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            adjacent = self.levels.get((key[0] + dx, key[1] + dy), ())
            if any(self._edge_ok(key, support) for support in adjacent):
                continue
            evidence = self.space.portal_evidence(
                x, y, elevation, dx, dy, self.grid_m,
                self.hard_traversal_clearance_m, self.portal_probe_range_m,
            )
            if evidence.is_frontier:
                directions.append((dx, dy))
                information += evidence.unknown_columns
        return tuple(directions), information, 0

    def _visited(self, key):
        z_cells = int(math.ceil(self.visited_vertical_tolerance_m / self.z_grid_m))
        for dx, dy in self.visit_offsets:
            levels = self.visited_index.get((key[0] + dx, key[1] + dy), ())
            if any(abs(level - key[2]) <= z_cells for level in levels):
                return True
        return False

    def _blocked(self, key):
        x, y = key[0] * self.grid_m, key[1] * self.grid_m
        z = self.cells[key][0]
        return any(
            (x - blocked_x) ** 2 + (y - blocked_y) ** 2 <= self.blocked_subgoal_radius_m ** 2
            and abs(z - blocked_z) <= self.visited_vertical_tolerance_m
            for blocked_x, blocked_y, blocked_z, _stamp in self.blocked_subgoals
        )

    def _waypoint_clearance(self, key):
        return self.space.obstacle_clearance(
            key[0] * self.grid_m,
            key[1] * self.grid_m,
            self.cells[key][0],
        )

    def _safe_waypoint(self, key, minimum_clearance_m=None):
        if self._blocked(key):
            return False
        minimum_clearance_m = (
            self.minimum_waypoint_clearance_m
            if minimum_clearance_m is None
            else max(self.hard_traversal_clearance_m, float(minimum_clearance_m))
        )
        x, y = key[0] * self.grid_m, key[1] * self.grid_m
        if self.space.column_state(
            x,
            y,
            self.cells[key][0],
            self.hard_traversal_clearance_m,
        ) == OCCUPIED:
            return False
        return self._waypoint_clearance(key) >= minimum_clearance_m

    def _prune_memories(self, now_s):
        self.blocked_subgoals = [
            item for item in self.blocked_subgoals if now_s - item[3] <= self.blocked_subgoal_memory_s
        ]
        self.portal_retry_after_s = {
            key: retry for key, retry in self.portal_retry_after_s.items() if retry > now_s
        }
        self.recovery_until_s = {
            key: retry for key, retry in self.recovery_until_s.items() if retry > now_s
        }

    def _clear_global_goal(self):
        self.active_goal = None
        self.active_goal_reached_latched = False
        self.global_goal = None
        self.frontier_anchor = None
        self.frontier_anchor_directions = ()
        self.frontier_unknown_before = 0
        self.global_goal_gain = 0
        self.global_goal_kind = ""
        self.global_coverage_before = 0
        self.global_coverage_revision = self.space.coverage_revision
        self.global_coverage_keys = ()
        self.global_region_support_keys = ()
        self.active_region = None
        self.global_goal_search_range_m = self.search_range_m
        self.global_goal_max_expansions = self.max_expansions
        self.global_goal_selected_s = None
        self.global_observe_started_s = None
        self.global_observe_revision = self.space.revision

    def _new_cell_count(self, key, since_s):
        if key is None or since_s is None:
            return 0
        radius_cells = int(math.ceil(self.frontier_progress_radius_m / self.grid_m))
        elevation = self.cells.get(key, (key[2] * self.z_grid_m,))[0]
        count = 0
        for ox in range(-radius_cells, radius_cells + 1):
            for oy in range(-radius_cells, radius_cells + 1):
                if math.hypot(ox * self.grid_m, oy * self.grid_m) > self.frontier_progress_radius_m:
                    continue
                for candidate in self.levels.get((key[0] + ox, key[1] + oy), ()):
                    if (
                        abs(self.cells[candidate][0] - elevation) <= self.frontier_cluster_vertical_m
                        and self.cell_first_seen_s.get(candidate, -float("inf")) > since_s
                    ):
                        count += 1
        return count

    def _support_point(self, key):
        return key[0] * self.grid_m, key[1] * self.grid_m, self.cells[key][0]

    def _missed_coverage_keys(self, viewpoint):
        # One graph cell in each direction covers small localization and
        # stopping variations. The level index keeps separate decks independent.
        ix, iy, iz = viewpoint
        missed = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                missed.update(
                    self.missed_coverage_by_viewpoint.get((ix + dx, iy + dy, iz), ())
                )
        return missed

    def _mark_frontier_region(self, radius_m, reason):
        if self.frontier_anchor is None or not self.frontier_anchor_directions:
            return
        elevation = self.cells.get(
            self.frontier_anchor,
            (self.frontier_anchor[2] * self.z_grid_m,),
        )[0]
        added = self.space.mark_completed(
            self.frontier_anchor[0] * self.grid_m,
            self.frontier_anchor[1] * self.grid_m,
            elevation,
            self.frontier_anchor_directions,
            radius_m,
        )
        self._log_info(
            "FLEX-E completed physical region at (%.2f, %.2f), radius=%.1f m, cells=%d, reason=%s",
            self.frontier_anchor[0] * self.grid_m,
            self.frontier_anchor[1] * self.grid_m,
            radius_m,
            added,
            reason,
        )

    def _fail_frontier(self, now_s, failed_subgoal, reason):
        if failed_subgoal is not None:
            elevation = self.cells.get(failed_subgoal, (failed_subgoal[2] * self.z_grid_m,))[0]
            self.blocked_subgoals.append(
                (failed_subgoal[0] * self.grid_m, failed_subgoal[1] * self.grid_m, elevation, now_s)
            )
        failed_region = self.active_region
        failures = self.region_tracker.defer(failed_region, now_s)
        if self.frontier_anchor is not None and failures >= self.region_tracker.max_failures:
            self.portal_retry_after_s[self.frontier_anchor] = now_s + self.blocked_subgoal_memory_s
        self.preferred_region = None
        self._log_warning(
            "FLEX-E physical region deferred: %s (region=%s failures=%d)",
            reason,
            failed_region,
            failures,
        )
        self._clear_global_goal()

    def _finish_frontier_observation(self, now_s):
        observed_region = self.active_region
        if self.global_goal_kind == "coverage":
            target_keys = tuple(self.global_coverage_keys)
            remaining = {
                key for key in target_keys if not self.space.is_support_key_covered(key)
            }
            covered_after = len(target_keys) - len(remaining)
            coverage_gain = max(0, covered_after - self.global_coverage_before)
            region_keys = tuple(self.global_region_support_keys)
            region_covered = sum(
                self.space.is_support_key_covered(key) for key in region_keys
            )
            total = len(region_keys)
            coverage_ratio = float(region_covered) / total if total else 0.0
            fresh_scan = self.space.revision > self.global_observe_revision
            if fresh_scan and self.global_observe_started_s is not None and remaining:
                self.missed_coverage_by_viewpoint.setdefault(
                    self.global_goal, set()
                ).update(remaining)
            if coverage_gain:
                outcome = "coverage_region_advanced"
                self.region_tracker.mark_progress(observed_region)
                self.preferred_region = (
                    observed_region
                    if remaining and coverage_gain >= self.coverage_min_observation_gain
                    else None
                )
            else:
                outcome = (
                    "coverage_viewpoint_no_information_gain"
                    if fresh_scan else "coverage_viewpoint_no_fresh_scan"
                )
                if not fresh_scan:
                    self.region_tracker.defer(observed_region, now_s)
                self.preferred_region = None
            self._log_info(
                "FLEX-E free-space observation: %s, target_covered=%d->%d/%d, gain=%d, missed=%d, region_coverage=%.3f, region=%s",
                outcome,
                self.global_coverage_before,
                covered_after,
                len(target_keys),
                coverage_gain,
                len(remaining),
                coverage_ratio,
                observed_region,
            )
            self._clear_global_goal()
            return outcome
        if self.frontier_anchor not in self.cells:
            self._clear_global_goal()
            return "frontier_observed_without_anchor"
        directions, unknown_after, _risk = self._frontier_evidence(
            self.frontier_anchor, include_completed=True
        )
        growth = self._new_cell_count(
            self.frontier_anchor, self.global_goal_selected_s
        )
        portal_directions, portal_unknown, _ = self._portal_frontier_evidence(
            self.frontier_anchor, now_s, ignore_cooldown=True
        )
        if self.global_goal_kind == "portal":
            if growth > 0 and (len(portal_directions) == 0 or growth >= self.frontier_min_growth_cells):
                outcome = "portal_observation_advanced"
                self.region_tracker.mark_progress(observed_region)
                self.preferred_region = observed_region
            else:
                outcome = "portal_observation_deferred"
                self.portal_retry_after_s[self.frontier_anchor] = now_s + self.blocked_subgoal_memory_s
                self.region_tracker.defer(observed_region, now_s)
                self.preferred_region = None
            self._log_info(
                "FLEX-E portal (%s) observation: %s, unknown=%d->%d, new_support=%d",
                self.frontier_anchor, outcome, self.frontier_unknown_before, portal_unknown, growth,
            )
            self._clear_global_goal()
            return outcome
        if not portal_directions and (
            not directions or unknown_after < max(1, self.frontier_unknown_before // 3)
        ):
            outcome = "frontier_region_closed"
            radius_m = self.completed_region_radius_m
            self.region_tracker.mark_progress(observed_region)
            self.preferred_region = observed_region
        elif growth >= self.frontier_min_growth_cells:
            outcome = "frontier_region_advanced"
            radius_m = 0.0 if portal_directions else self.completed_region_progress_radius_m
            self.region_tracker.mark_progress(observed_region)
            self.preferred_region = observed_region
        else:
            outcome = "frontier_region_no_information_gain"
            radius_m = 0.0
            self.region_tracker.defer(observed_region, now_s)
            self.preferred_region = None
        if radius_m > 0.0:
            self._mark_frontier_region(radius_m, outcome)
        self._log_info(
            "FLEX-E frontier observation: %s, unknown=%d->%d, new_support=%d",
            outcome,
            self.frontier_unknown_before,
            unknown_after,
            growth,
        )
        self._clear_global_goal()
        return outcome

    def _dijkstra(self, root, search_range_m, max_expansions):
        queue, costs, parents = [(0.0, root)], {root: 0.0}, {root: None}
        beyond_range = set()
        expanded = 0
        while queue and expanded < max_expansions:
            cost, key = heapq.heappop(queue)
            if cost != costs.get(key) or cost > search_range_m:
                continue
            expanded += 1
            neighbors = list(self._neighbors(key))
            for neighbor in neighbors:
                edge = self.grid_m * math.hypot(neighbor[0] - key[0], neighbor[1] - key[1])
                candidate = cost + edge + 0.5 * abs(self.cells[neighbor][0] - self.cells[key][0])
                if candidate >= costs.get(neighbor, float("inf")):
                    continue
                if candidate > search_range_m:
                    beyond_range.add(neighbor)
                    continue
                costs[neighbor], parents[neighbor] = candidate, key
                heapq.heappush(queue, (candidate, neighbor))
        node_limit = expanded >= max_expansions and any(
            cost == costs.get(key) and cost <= search_range_m for cost, key in queue
        )
        truncated = node_limit or any(key not in costs for key in beyond_range)
        return parents, costs, expanded, truncated

    def _cluster_frontiers(self, items):
        if not items:
            return []
        by_key = {item[1]: item for item in items}
        by_xy = defaultdict(list)
        for key in by_key:
            by_xy[key[:2]].append(key)
        pending = set(by_key)
        link_cells = int(math.ceil(self.frontier_cluster_link_m / self.grid_m))
        clusters = []
        while pending:
            seed = min(pending)
            pending.remove(seed)
            stack, keys = [seed], [seed]
            while stack:
                key = stack.pop()
                for ox in range(-link_cells, link_cells + 1):
                    for oy in range(-link_cells, link_cells + 1):
                        for candidate in by_xy.get((key[0] + ox, key[1] + oy), ()):
                            if candidate not in pending:
                                continue
                            if abs(self.cells[candidate][0] - self.cells[key][0]) > self.frontier_cluster_vertical_m:
                                continue
                            pending.remove(candidate)
                            stack.append(candidate)
                            keys.append(candidate)
            clusters.append([by_key[key] for key in keys])
        return clusters

    def _region_for_cell(self, key):
        return self.region_tracker.region_key(*self._support_point(key))

    def _connected_coverage_gaps(self, uncovered):
        """Find groups of unobserved support joined by safe terrain edges."""
        pending = set(uncovered)
        gaps = {}
        while pending:
            key = pending.pop()
            group, stack = {key}, [key]
            while stack:
                for neighbor in self._neighbors(stack.pop()):
                    if neighbor in pending:
                        pending.remove(neighbor)
                        group.add(neighbor)
                        stack.append(neighbor)
            unique_support = frozenset(uncovered[key] for key in group)
            if len(unique_support) >= self.coverage_patch_min_cells:
                gaps.update((key, unique_support) for key in unique_support)
        return gaps

    def _coverage_region_choice(
        self,
        region,
        uncovered_points,
        reachable_by_region,
        region_support_keys,
        costs,
        root,
        vehicle,
        min_goal_distance_m,
        allow_root_viewpoint,
    ):
        """Choose a high-clearance interior viewpoint for free-space coverage."""

        points = list(uncovered_points)[: self.coverage_max_points_per_region]
        region_cells = list(reachable_by_region.get(region, ()))
        if not points or not region_cells:
            return None
        # Sample around the missing support, including across region
        # boundaries. Use an interior pool as an alternative approach.
        near_cells = set(region_cells)
        samples = points[::max(1, math.ceil(len(points) / 8))]
        radius_cells = int(math.ceil(2.0 / self.grid_m))
        for x, y, _ in samples:
            ix, iy = round(x / self.grid_m), round(y / self.grid_m)
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    near_cells.update(
                        key for key in self.levels.get((ix + dx, iy + dy), ())
                        if key in costs and self._region_for_cell(key)[2] == region[2]
                    )
        ground = {key: self._support_point(key) for key in near_cells}
        gap_distance = {
            key: min((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2
                     for px, py, pz in samples)
            for key, (x, y, z) in ground.items()
        }
        limit = self.coverage_candidate_limit
        near_pool = sorted(
            near_cells, key=lambda key: (gap_distance[key], costs[key], key)
        )[:2 * limit]
        near_pool = sorted(
            near_pool,
            key=lambda key: (
                gap_distance[key] / 4.0 - min(2.0, self._waypoint_clearance(key)),
                costs[key],
                key,
            ),
        )[:limit]
        interior_pool = sorted(
            region_cells, key=lambda key: (-self._waypoint_clearance(key), costs[key], key)
        )[:limit // 2]
        candidates = dict.fromkeys(near_pool + interior_pool)

        best = None
        for key in candidates:
            if key == root and not allow_root_viewpoint:
                continue
            if not self._safe_waypoint(key):
                continue
            x, y, ground_z = ground[key]
            direct = math.hypot(x - vehicle[0], y - vehicle[1])
            if direct < min_goal_distance_m:
                continue
            if len(list(self._neighbors(key))) < self.min_frontier_support:
                continue
            visibility_keys = self.space.support_visibility_keys(
                (x, y, ground_z),
                points,
                self.vehicle_height_m,
                self.coverage_max_points_per_region,
                self.coverage_prediction_min_known_fraction,
                excluded_keys=self._missed_coverage_keys(key),
            )
            visibility_gain = len(visibility_keys)
            if visibility_gain < min(self.coverage_min_viewpoint_gain, len(points)):
                continue
            cost = costs.get(key, float("inf"))
            target_heading = math.atan2(y - vehicle[1], x - vehicle[0])
            heading_alignment = math.cos(target_heading - self.vehicle_yaw)
            clearance = self._waypoint_clearance(key)
            score = (
                self.coverage_gain_reward * math.log1p(visibility_gain)
                + self.clearance_reward * min(clearance, self.space.clearance_search_m)
                + 1.0 / (1.0 + math.sqrt(gap_distance[key]))
                + self.distance_reward * cost
                + 0.5 * self.heading_reward * heading_alignment
                - self.visited_penalty * int(self._visited(key))
            )
            choice = {
                "score": score,
                "anchor": key,
                "viewpoint": key,
                "gain": 0,
                "information_gain": visibility_gain,
                "risk": 0,
                "cost": cost,
                "directions": (),
                "kind": "coverage",
                "region": region,
                "uncovered_count": len(uncovered_points),
                "priority": 0,
                "coverage_keys": visibility_keys,
                "region_support_keys": tuple(sorted(region_support_keys)),
            }
            if best is None or (choice["score"], key) > (best["score"], best["anchor"]):
                # A cleanup view should retain a usable local exit under the
                # current footprint check, including when it is across a tile.
                _, exit_support, _, _ = self._dijkstra(key, 3.0, 8)
                if len(exit_support) >= 8:
                    best = choice
        return best

    def _publish_region_debug(self):
        if not self.publish_debug_clouds or self._debug_callback is None:
            return
        status_value = {
            EXPLORING: 1.0, DEFERRED: 2.0,
            COVERED: 3.0, INACCESSIBLE: 4.0,
        }
        points = []
        for key, state in sorted(self.region_tracker.regions.items()):
            if state.status in status_value:
                points.append(
                    self.region_tracker.region_center(key)
                    + (status_value[state.status],)
                )
        self._debug_callback("regions", points)

    def _publish_frontier_debug(self, items, viewpoints):
        if not self.publish_debug_clouds or self._debug_callback is None:
            return
        frontier_points = [
            (item[1][0] * self.grid_m, item[1][1] * self.grid_m, self.cells[item[1]][0])
            for item in items
        ]
        viewpoint_points = [
            (key[0] * self.grid_m, key[1] * self.grid_m, self.cells[key][0])
            for key in viewpoints
        ]
        self._debug_callback("frontiers", frontier_points)
        self._debug_callback("viewpoints", viewpoint_points)

    def _select_frontier(
        self,
        root,
        vehicle,
        now_s,
        search_range_m=None,
        max_expansions=None,
        min_goal_distance_m=None,
        allow_root_viewpoint=False,
        resolve_orphans=False,
        frontier_only=False,
    ):
        self._prune_memories(now_s)
        search_range_m = self.search_range_m if search_range_m is None else search_range_m
        max_expansions = self.max_expansions if max_expansions is None else max_expansions
        min_goal_distance_m = (
            self.goal_min_distance_m
            if min_goal_distance_m is None
            else min_goal_distance_m
        )
        parents, costs, expanded, truncated = self._dijkstra(
            root, search_range_m, max_expansions
        )
        if resolve_orphans and truncated and expanded < max_expansions:
            # The distance budget stopped a final audit before the node
            # budget did. Follow the remaining connected graph before
            # declaring a region disconnected or the component closed.
            search_range_m = float("inf")
            parents, costs, expanded, truncated = self._dijkstra(
                root, search_range_m, max_expansions
            )
        required_reachable = self._minimum_audit_reachable()
        valid_attachment = resolve_orphans and self._valid_audit_attachment(costs)
        if resolve_orphans and not truncated and not valid_attachment:
            radius_cells = int(math.ceil(1.5 / self.grid_m))
            reattached = self._connected_recovery_anchor(
                vehicle, self._attachment_candidates(vehicle, radius_cells), costs, now_s
            )
            if reattached is not None:
                self._log_info("FLEX-E reattached graph root %s -> %s", root, reattached)
                root = reattached
                parents, costs, expanded, truncated = self._dijkstra(
                    root, float("inf"), max_expansions
                )
                search_range_m = float("inf")
                valid_attachment = self._valid_audit_attachment(costs)
        detached = resolve_orphans and not valid_attachment
        full_audit = resolve_orphans and not truncated and not detached
        # Keep the largest verified component as the attachment reference.
        # Otherwise several smaller audits could lower the guard step by step
        # and eventually certify an isolated fragment.
        if full_audit and len(costs) >= len(self.last_healthy_component):
            self.last_healthy_component = set(costs)
        reachable_by_region = defaultdict(list)
        for key in costs:
            reachable_by_region[self._region_for_cell(key)].append(key)
        observed_regions = set(reachable_by_region)
        region_evidence = defaultdict(
            lambda: {
                "frontier_points": 0,
                "covered_cells": 0,
                "total_cells": 0,
                "selectable": 0,
            }
        )
        support_by_region = {}
        uncovered_by_region = defaultdict(dict)
        uncovered = {}
        for region, keys in reachable_by_region.items():
            evidence = region_evidence[region]
            support = set()
            for key in keys:
                point = self._support_point(key)
                if self.space.column_state(*point) == OCCUPIED:
                    continue
                support_key = self.space.support_key(*point)
                covered = self.space.is_support_key_covered(support_key)
                if not covered:
                    uncovered_by_region[region].setdefault(support_key, point)
                    uncovered[key] = support_key
                elif support_key not in support:
                    evidence["covered_cells"] += 1
                support.add(support_key)
            support_by_region[region] = support
            evidence["total_cells"] = len(support)
        gaps = (
            self._connected_coverage_gaps(uncovered)
            if not frontier_only and self.coverage_patch_min_cells else {}
        )
        if not frontier_only and self.coverage_patch_min_cells:
            if full_audit:
                self.pending_gap_keys.difference_update(self.space.covered_support)
                self.pending_gap_keys.update(gaps)
            for support_key in set(uncovered.values()).intersection(self.pending_gap_keys):
                gaps.setdefault(support_key, frozenset((support_key,)))
        gap_points = {
            support_key: self._support_point(key)
            for key, support_key in uncovered.items() if support_key in gaps
        }
        items = []
        portal_anchors = set()
        for key, cost in costs.items():
            if key == root or cost > search_range_m:
                continue
            directions, information_gain, risk = self._frontier_evidence(key)
            if not directions and resolve_orphans:
                directions, information_gain, risk = self._portal_frontier_evidence(
                    key, now_s, ignore_cooldown=True
                )
                if directions:
                    portal_anchors.add(key)
            gain = len(directions)
            if not (0 < gain <= self.max_frontier_gain):
                continue
            neighbors = list(self._neighbors(key))
            if len(neighbors) < self.min_frontier_support:
                continue
            x, y = key[0] * self.grid_m, key[1] * self.grid_m
            direct = math.hypot(x - vehicle[0], y - vehicle[1])
            target_heading = math.atan2(y - vehicle[1], x - vehicle[0])
            heading_alignment = math.cos(target_heading - self.vehicle_yaw)
            score = (
                2.0 * math.log1p(information_gain)
                + 0.25 * min(8, len(neighbors))
                + self.distance_reward * cost
                + self.heading_reward * heading_alignment
                - 0.20 * risk
                + self.rng.uniform(-1.0e-4, 1.0e-4)
            )
            if key == self.last_goal:
                score -= 3.0
            region = self._region_for_cell(key)
            region_evidence[region]["frontier_points"] += 1
            items.append(
                (score, key, gain, information_gain, risk, cost, direct, directions, region)
            )

        clusters = self._cluster_frontiers(items)
        choices, debug_viewpoints = [], []
        for cluster in clusters:
            candidates = []
            for item in cluster:
                score, key, gain, information_gain, risk, cost, direct, directions, region = item
                if key in portal_anchors and now_s < self.portal_retry_after_s.get(key, 0.0):
                    continue
                if direct < min_goal_distance_m:
                    continue
                viewpoint = None
                for option in self._frontier_viewpoint_options(
                    root, key, parents, costs, at_frontier=key in portal_anchors
                ):
                    if option == root and not allow_root_viewpoint:
                        continue
                    if key in portal_anchors and option == root:
                        continue
                    if not self._safe_waypoint(
                        option,
                        self.minimum_frontier_waypoint_clearance_m,
                    ):
                        continue
                    if self.space.has_clear_line_of_sight(
                        (
                            option[0] * self.grid_m,
                            option[1] * self.grid_m,
                            self.cells[option][0],
                        ),
                        (
                            key[0] * self.grid_m,
                            key[1] * self.grid_m,
                            self.cells[key][0],
                        ),
                        self.hard_traversal_clearance_m,
                    ):
                        viewpoint = option
                        break
                if viewpoint is None:
                    continue
                repeat_penalty = self.visited_penalty * (
                    int(self._visited(key)) + int(self._visited(viewpoint))
                )
                candidates.append(
                    (
                        score - repeat_penalty,
                        key,
                        viewpoint,
                        gain,
                        information_gain,
                        risk,
                        cost,
                        directions,
                        region,
                    )
                )
            if not candidates:
                continue
            best = max(candidates)
            (
                candidate_score,
                key,
                viewpoint,
                gain,
                information_gain,
                risk,
                cost,
                directions,
                region,
            ) = best
            choices.append(
                {
                    "score": candidate_score
                    + self.frontier_cluster_gain_reward * math.sqrt(len(cluster)),
                    "anchor": key,
                    "viewpoint": viewpoint,
                    "gain": gain,
                    "information_gain": information_gain,
                    "risk": risk,
                    "cost": cost,
                    "directions": directions,
                    "kind": "portal" if key in portal_anchors else "frontier",
                    "region": region,
                    "uncovered_count": len(uncovered_by_region.get(region, ())),
                    "priority": 1,
                    "coverage_keys": (),
                    "region_support_keys": (),
                }
            )
            region_evidence[region]["selectable"] += 1
            debug_viewpoints.append(best[2])

        # Finish the interior of an entered region from a high-clearance
        # viewpoint. Obstacle surfaces themselves never become goals.
        for region, unobserved in uncovered_by_region.items():
            evidence = region_evidence[region]
            total = evidence["total_cells"]
            covered = evidence["covered_cells"]
            coverage_ratio = float(covered) / total if total else 0.0
            if frontier_only:
                # The uncovered support keeps this region open until the
                # later cleanup phase searches for an actual viewpoint.
                continue
            local_gap_keys = set(unobserved).intersection(gaps)
            evidence["gap_cells"] = len(local_gap_keys)
            sparse_gap = (
                len(unobserved) < self.region_tracker.coverage_min_uncovered_cells
                or coverage_ratio >= self.region_tracker.coverage_complete_ratio
            )
            if sparse_gap and not local_gap_keys:
                continue
            patch_groups = {gaps[key] for key in local_gap_keys}
            patch_keys = {key for group in patch_groups for key in group}
            points = [gap_points[key] for key in sorted(patch_keys)]
            if not sparse_gap:
                points.extend(point for key, point in unobserved.items() if key not in patch_keys)
            choice = self._coverage_region_choice(
                region,
                points,
                reachable_by_region,
                support_by_region[region],
                costs,
                root,
                vehicle,
                min_goal_distance_m,
                allow_root_viewpoint,
            )
            if choice is None:
                continue
            region_evidence[region]["selectable"] += 1
            choices.append(choice)
            debug_viewpoints.append(choice["viewpoint"])

        if not detached:
            self.region_tracker.update(
                region_evidence, observed_regions, now_s, allow_closure=full_audit
            )
        resolved = []
        # Only a complete reachability pass can prove that a historical
        # region is disconnected.  A node-limited audit must not quarantine
        # the unexplored tail of its own truncated search.
        if full_audit:
            resolved.extend(self.region_tracker.resolve_unselectable(observed_regions, now_s))
            resolved.extend(
                self.region_tracker.resolve_orphans(observed_regions, now_s)
            )
        for region, status in resolved:
            self._log_info(
                "FLEX-E resolved unselectable region=%s as %s",
                region,
                status,
            )
        self._publish_region_debug()
        counts = self.region_tracker.status_counts(observed_regions)
        reachable_unresolved = self.region_tracker.unresolved_count(
            observed_regions, now_s
        )
        all_unresolved = self.region_tracker.unresolved_count(now_s=now_s)
        orphaned_unresolved = max(0, all_unresolved - reachable_unresolved)
        stats = {
            "raw": len(items),
            "regions": len(clusters),
            "selectable": len(choices),
            "unresolved": reachable_unresolved
            + (orphaned_unresolved if resolve_orphans else 0),
            "orphaned": orphaned_unresolved,
            "reachable": len(costs),
            "minimum_reachable": required_reachable,
            "root": root,
            "valid_attachment": valid_attachment,
            "truncated": int(truncated),
            "search_range_m": search_range_m,
            "uncovered_cells": sum(
                max(0, values["total_cells"] - values["covered_cells"])
                for values in region_evidence.values()
            ),
            "gap_cells": len(gaps),
            "exploring_regions": counts[EXPLORING],
            "covered_regions": counts[COVERED],
            "deferred_regions": counts[DEFERRED],
            "inaccessible_regions": counts[INACCESSIBLE],
            "route_regions": 0,
            "selected_uncovered_count": 0,
            "selected_coverage_ratio": 1.0,
            "selected_coverage_keys": (),
            "selected_region_support_keys": (),
        }
        self._publish_frontier_debug(items, debug_viewpoints)
        active_choices = [
            choice
            for choice in choices
            if not detached and self.region_tracker.status(choice["region"], now_s) == EXPLORING
        ]
        frontier_choices = [
            choice for choice in active_choices if choice["kind"] in ("frontier", "portal")
        ]
        coverage_choices = [
            choice for choice in active_choices if choice["kind"] == "coverage"
        ]
        # Expansion is a strict first phase.  Interior coverage cannot hide a
        # doorway merely because it is nearer: use coverage cleanup only after
        # the current search scope contains no executable real frontier.
        eligible_choices = (
            frontier_choices
            if frontier_only or frontier_choices
            else coverage_choices
        )
        stats["selectable_frontiers"] = len(frontier_choices)
        stats["selectable_coverage"] = len(coverage_choices)
        best_by_region = {}
        for choice in eligible_choices:
            previous = best_by_region.get(choice["region"])
            if previous is None or (
                choice["priority"], choice["score"], choice["anchor"]
            ) > (
                previous["priority"], previous["score"], previous["anchor"]
            ):
                best_by_region[choice["region"]] = choice
        stats["selectable"] = len(best_by_region)
        if self.frontier_gain_rate_selection or (coverage_choices and not frontier_choices):
            def priority(region):
                choice = best_by_region[region]
                path_cost = costs[choice["viewpoint"]]
                gain = math.sqrt(max(1, choice["information_gain"]))
                revisit_factor = 0.75 if self._visited(choice["viewpoint"]) else 1.0
                continuity = 1.15 if region == self.preferred_region else 1.0
                # Fixed approach/sensing cost avoids chasing tiny nearby gains.
                rate = continuity * revisit_factor * gain / (8.0 + path_cost)
                return -rate, path_cost, region

            self.planned_region_route = sorted(best_by_region, key=priority)
        else:
            route_input = {
                region: (self._key_to_pose(choice["viewpoint"]), choice["cost"])
                for region, choice in best_by_region.items()
            }
            self.planned_region_route = self.region_tracker.route(
                route_input, vehicle[:3], self.preferred_region,
            )
        stats["route_regions"] = len(self.planned_region_route)
        if self.planned_region_route:
            selected_region = self.planned_region_route[0]
            best = best_by_region[selected_region]
            self.region_tracker.mark_selected(selected_region, now_s)
            stats["selected_uncovered_count"] = best["uncovered_count"]
            stats["selected_coverage_keys"] = best["coverage_keys"]
            stats["selected_region_support_keys"] = best["region_support_keys"]
            state = self.region_tracker.regions.get(selected_region)
            stats["selected_coverage_ratio"] = (
                state.coverage_ratio if state is not None else 0.0
            )
            return (
                best["anchor"],
                best["viewpoint"],
                best["gain"],
                best["information_gain"],
                best["directions"],
                expanded,
                parents,
                costs,
                stats,
                selected_region,
                best["kind"],
            )
        return None, None, 0, 0, (), expanded, parents, costs, stats, None, ""

    def _path_to_goal(self, root, target, now_s, search_range_m, max_expansions):
        if target not in self.cells:
            return None, None, 0
        self._prune_memories(now_s)

        def estimate(key):
            horizontal = self.grid_m * math.hypot(target[0] - key[0], target[1] - key[1])
            return horizontal + 0.5 * abs(self.cells[target][0] - self.cells[key][0])

        queue, costs, parents = [(estimate(root), 0.0, root)], {root: 0.0}, {root: None}
        expanded = 0
        while queue and expanded < max_expansions:
            _, cost, key = heapq.heappop(queue)
            if cost != costs.get(key) or cost > search_range_m:
                continue
            expanded += 1
            if key == target:
                return parents, costs, expanded
            for neighbor in self._neighbors(key):
                edge = self.grid_m * math.hypot(neighbor[0] - key[0], neighbor[1] - key[1])
                candidate = cost + edge + 0.5 * abs(self.cells[neighbor][0] - self.cells[key][0])
                if candidate < costs.get(neighbor, float("inf")) and candidate <= search_range_m:
                    costs[neighbor], parents[neighbor] = candidate, key
                    heapq.heappush(queue, (candidate + estimate(neighbor), candidate, neighbor))
        return None, None, expanded

    def _route_prefix(
        self,
        root,
        target,
        parents,
        costs,
        minimum_clearance_m=None,
    ):
        chain, cursor = [], target
        while cursor is not None:
            chain.append(cursor)
            cursor = parents.get(cursor)
        if not chain or chain[-1] != root:
            return target
        choice = root
        for key in reversed(chain[:-1]):
            if costs[key] > self.execution_horizon_m + 1.0e-6:
                break
            if self._safe_waypoint(key, minimum_clearance_m):
                choice = key
        if choice != root or len(chain) == 1:
            return choice
        for key in reversed(chain[:-1]):
            if self._safe_waypoint(key, minimum_clearance_m):
                return key
        return root

    def _egress_prefix(
        self,
        root,
        target,
        parents,
        costs,
        source_region,
        minimum_clearance_m=None,
    ):
        """Place a transit waypoint beyond the exit of a completed region.

        Stopping exactly on a narrow threshold makes the local controller
        oscillate. When the route leaves the current physical subspace within
        the execution horizon, continue to the first high-clearance support
        cell at least ``portal_exit_distance_m`` beyond that crossing.
        """

        chain, cursor = [], target
        while cursor is not None:
            chain.append(cursor)
            cursor = parents.get(cursor)
        if not chain or chain[-1] != root:
            return self._route_prefix(
                root,
                target,
                parents,
                costs,
                minimum_clearance_m,
            )
        ordered = list(reversed(chain))

        # Detect the actual low-clearance part of the route first. This works
        # even when a doorway and both rooms happen to lie in the same coarse
        # 8 m subspace. A narrow cell remains path-traversable because the
        # inflated occupancy check has already accepted it; it is simply not
        # a suitable place for the controller to stop and turn.
        narrow_seen = False
        last_narrow_cost = 0.0
        for key in ordered:
            cost = costs.get(key, float("inf"))
            if cost > self.portal_exit_search_m + 1.0e-6:
                break
            x, y = key[0] * self.grid_m, key[1] * self.grid_m
            passable = (
                not self._blocked(key)
                and self.space.column_state(
                    x,
                    y,
                    self.cells[key][0],
                    self.hard_traversal_clearance_m,
                )
                != OCCUPIED
            )
            if passable and self._waypoint_clearance(key) < self.minimum_waypoint_clearance_m:
                narrow_seen = True
                last_narrow_cost = cost
                continue
            if (
                narrow_seen
                and self._safe_waypoint(key)
                and cost - last_narrow_cost >= self.portal_exit_distance_m
            ):
                return key

        transition_index = None
        for index, key in enumerate(ordered[1:], 1):
            if self._region_for_cell(key) != source_region:
                transition_index = index
                break
        if transition_index is None:
            return self._route_prefix(
                root,
                target,
                parents,
                costs,
                minimum_clearance_m,
            )
        transition_cost = costs.get(ordered[transition_index], float("inf"))
        if transition_cost > self.portal_exit_search_m + 1.0e-6:
            return self._route_prefix(
                root,
                target,
                parents,
                costs,
                minimum_clearance_m,
            )
        fallback = None
        maximum_cost = transition_cost + self.portal_exit_search_m
        for key in ordered[transition_index:]:
            cost = costs.get(key, float("inf"))
            if cost > maximum_cost + 1.0e-6:
                break
            if not self._safe_waypoint(key):
                continue
            fallback = key
            if cost - transition_cost >= self.portal_exit_distance_m:
                return key
        return fallback or self._route_prefix(
            root,
            target,
            parents,
            costs,
            minimum_clearance_m,
        )

    def _frontier_viewpoint_options(self, root, target, parents, costs, at_frontier=False):
        """Order observed support cells by proximity to the desired standoff."""

        chain, cursor = [], target
        while cursor is not None:
            chain.append(cursor)
            cursor = parents.get(cursor)
        if not chain or chain[-1] != root:
            return (target,)
        target_cost = costs[target]
        standoff = 0.0 if at_frontier else self.frontier_standoff_m
        return tuple(
            sorted(
                chain,
                key=lambda key: (
                    abs((target_cost - costs.get(key, 0.0)) - standoff),
                    -costs.get(key, 0.0),
                ),
            )
        )

    def _global_arrival_radius(self):
        if self.global_goal_kind == "portal":
            return min(self.global_goal_reached_m, 0.60)
        if self.global_goal_kind == "coverage":
            return min(self.global_goal_reached_m, 0.75)
        return self.global_goal_reached_m

    def _active_arrival_radius(self):
        if self.global_goal_kind in ("bootstrap", "recovery"):
            return min(self.goal_reached_m, 0.45)
        if self.global_goal_kind in ("portal", "coverage") and self.active_goal == self.global_goal:
            return min(self.goal_reached_m, self._global_arrival_radius())
        return self.goal_reached_m

    def _active_goal_state(self, now_s):
        if self.active_goal is None or self.vehicle is None:
            return "none"
        if self.active_goal not in self.cells:
            self.active_goal = None
            self.active_goal_reached_latched = False
            return "invalid"
        x, y = self.active_goal[0] * self.grid_m, self.active_goal[1] * self.grid_m
        goal_z = self.cells[self.active_goal][0] + self.vehicle_height_m
        horizontal = math.hypot(x - self.vehicle[0], y - self.vehicle[1])
        dz = abs(goal_z - self.vehicle[2])
        distance = math.hypot(horizontal, dz)
        if self.active_goal_reached_latched or (
            horizontal <= self._active_arrival_radius() and dz <= self.goal_z_tolerance_m
        ):
            self.active_goal = None
            self.active_goal_reached_latched = False
            return "reached"
        if self.active_best_distance_m is None or distance < self.active_best_distance_m - self.progress_epsilon_m:
            self.active_best_distance_m, self.active_last_progress_s = distance, now_s
        if now_s - self.active_started_s < self.goal_hold_s:
            return "hold"
        if now_s - self.active_started_s >= self.goal_timeout_s:
            self.active_goal = None
            self.active_goal_reached_latched = False
            return "timeout"
        if now_s - self.active_last_progress_s < self.goal_stall_s:
            return "hold"
        self.active_goal = None
        self.active_goal_reached_latched = False
        return "stalled"

    def _publish_subgoal(self, target, final_goal, gain, now_s, row, reason):
        waypoint = self._key_to_pose(target)
        self.active_goal = target
        self.subgoal_revision += 1
        self.active_goal_reached_latched = False
        self.active_started_s = self.active_last_progress_s = now_s
        self.active_best_distance_m = math.hypot(
            math.hypot(waypoint[0] - self.vehicle[0], waypoint[1] - self.vehicle[1]),
            waypoint[2] - self.vehicle[2],
        )
        row.update(
            {
                "target_x": waypoint[0],
                "target_y": waypoint[1],
                "target_z": waypoint[2],
                "reason": reason,
            }
        )
        self._log_info(
            "FLEX-E %s subgoal=(%.2f, %.2f) global=(%.2f, %.2f) gain=%d",
            reason,
            waypoint[0],
            waypoint[1],
            final_goal[0] * self.grid_m,
            final_goal[1] * self.grid_m,
            gain,
        )

    def _finish_exploration(self, reason, now_s, row):
        if self.finished:
            return
        self.finished = True
        self._finish_reason = reason
        self._clear_global_goal()
        if self.vehicle is not None:
            row.update(
                {
                    "target_x": self.vehicle[0],
                    "target_y": self.vehicle[1],
                    "target_z": self.vehicle[2],
                }
            )
        self._closure_closed = reason == "reachable_space_closed"
        row["reason"] = "exploration_finished_" + reason
        self._log_info("FLEX-E exploration finished: %s at %.1f s", reason, now_s)

    def _key_to_pose(self, key):
        if key is None:
            return None
        cell = self.cells.get(key)
        elevation = cell[0] if cell is not None else key[2] * self.z_grid_m
        return (
            key[0] * self.grid_m,
            key[1] * self.grid_m,
            elevation + self.vehicle_height_m,
        )

    @staticmethod
    def _clock(now_s):
        return time.time() if now_s is None else float(now_s)

    def _log_info(self, fmt, *args):
        self._logger.info(fmt, *args)

    def _log_warning(self, fmt, *args):
        self._logger.warning(fmt, *args)

    def _log_info_throttle(self, interval, fmt, *args):
        self._log_throttle("info", interval, fmt, *args)

    def _log_warning_throttle(self, interval, fmt, *args):
        self._log_throttle("warning", interval, fmt, *args)

    def _log_throttle(self, level, interval, fmt, *args):
        now_s = time.monotonic()
        last_s = self._log_throttles.get((level, fmt), -float("inf"))
        if now_s - last_s >= interval:
            self._log_throttles[(level, fmt)] = now_s
            getattr(self._logger, level)(fmt, *args)

    def reset(self):
        """Clear maps, coverage, visit history, and active goals."""
        params, logger, debug_callback, param_source = (
            self._params.copy(), self._logger, self._debug_callback, self._param_source
        )
        self._close_trace()
        self.__init__(params, logger, debug_callback, param_source)

    def close(self):
        """Flush and close the optional decision trace."""
        with self.state_lock:
            self._close_trace()

    def report_failure(self, reason, now_s=None):
        """Defer the current physical region after a controller failure."""
        stamp = self._clock(now_s)
        with self.state_lock:
            if self.active_goal is None and self.global_goal is None:
                return False
            self._fail_frontier(stamp, self.active_goal, str(reason))
            self.next_frontier_search_s = 0.0
            return True

    def step(self, now_s=None):
        """Advance the planner and return the currently executable decision."""
        stamp = self._clock(now_s)
        with self.state_lock:
            self._plan_locked(stamp)
            subgoal = None if self.finished else self._key_to_pose(self.active_goal)
            viewpoint = self._key_to_pose(self.global_goal)
            anchor = self._key_to_pose(self.frontier_anchor)
            if viewpoint is not None and anchor is not None:
                heading_rad = math.atan2(
                    anchor[1] - viewpoint[1], anchor[0] - viewpoint[0]
                )
            elif viewpoint is not None and self.vehicle is not None:
                heading_rad = math.atan2(
                    viewpoint[1] - self.vehicle[1],
                    viewpoint[0] - self.vehicle[0],
                )
            else:
                heading_rad = self.vehicle_yaw
            if self.finished:
                status = "finished"
            elif (
                self.vehicle is None
                or len(self.cells) < 20
                or len(self.space.evidence) < self.visibility_min_voxels
            ):
                status = "waiting"
            elif self.global_observe_started_s is not None:
                status = "observing"
            elif self.active_goal is not None or self.global_goal is not None:
                status = "moving"
            else:
                status = "searching"
            diagnostics = dict(self._last_row)
            diagnostics["closure_closed"] = self._closure_closed
            diagnostics["finish_reason"] = self._finish_reason
            diagnostics["graph_revision"] = self.graph_revision
            diagnostics["occupancy_revision"] = self.space.revision
            diagnostics["subgoal_revision"] = self.subgoal_revision
            return {
                "status": status,
                "subgoal": subgoal,
                "global_viewpoint": viewpoint,
                "anchor": anchor,
                "kind": self.global_goal_kind,
                "heading_rad": heading_rad,
                "diagnostics": diagnostics,
            }

    def _plan_locked(self, now_s):
        started = time.perf_counter()
        row = {
            "sim_time_s": now_s,
            "cell_count": len(self.cells),
            "occupancy_voxels": len(self.space.evidence),
            "completed_region_cells": len(self.space.completed_regions),
            "expanded": 0,
            "frontiers": 0,
            "frontier_regions": 0,
            "uncovered_support_cells": 0,
            "connected_gap_cells": 0,
            "unresolved_regions": 0,
            "coverage_ratio": 1.0,
            "exploring_regions": 0,
            "covered_regions": 0,
            "deferred_regions": 0,
            "inaccessible_regions": 0,
            "orphaned_regions": 0,
            "global_region_route": 0,
            "selected_region": "",
            "target_kind": "",
            "reachable_cells": 0,
            "audit_root": "",
            "minimum_audit_reachable": self._minimum_audit_reachable(),
            "recovery_attempts": self.recovery_attempts,
            "search_truncated": 0,
            "closure_state": "",
            "information_gain": 0,
            "target_x": "",
            "target_y": "",
            "target_z": "",
            "reason": "",
        }
        self._last_row = row
        if self.finished:
            return
        if (
            self.vehicle is None
            or len(self.cells) < 20
            or len(self.space.evidence) < self.visibility_min_voxels
        ):
            row["reason"] = "waiting_for_pose_terrain_and_occupancy"
            row["decision_ms"] = 1000.0 * (time.perf_counter() - started)
            if now_s - self.last_idle_trace_s >= self.trace_idle_period_s:
                self.last_idle_trace_s = now_s
                self._trace(row)
            return
        if self.exploration_started_s is None:
            self.exploration_started_s = now_s
        elapsed_s = now_s - self.exploration_started_s
        if self.max_exploration_s > 0.0 and elapsed_s >= self.max_exploration_s:
            self._finish_exploration("time_limit", now_s, row)
            row["decision_ms"] = 1000.0 * (time.perf_counter() - started)
            self._last_runtime_s = row["decision_ms"] / 1000.0
            self._trace(row)
            self._timer_stop_requested = True
            return
        if (
            self.global_goal_kind == "coverage"
            and self.space.coverage_revision > self.global_coverage_revision
            and self.global_coverage_keys
            and all(self.space.is_support_key_covered(key) for key in self.global_coverage_keys)
        ):
            if self.active_goal is not None:
                self.subgoal_revision += 1
            row["reason"] = self._finish_frontier_observation(now_s)
        prior_subgoal = self.active_goal
        goal_state = self._active_goal_state(now_s)
        if (
            goal_state == "hold"
            and self.global_goal_kind == "recovery"
            and self.recovery_started_s is not None
            and now_s - self.recovery_started_s >= self.recovery_timeout_s
        ):
            self._clear_global_goal()
            goal_state = "none"
        if goal_state == "hold":
            row["reason"] = "holding_observed_goal"
            row["decision_ms"] = 1000.0 * (time.perf_counter() - started)
            if now_s - self.last_idle_trace_s >= self.trace_idle_period_s:
                self.last_idle_trace_s = now_s
                self._trace(row)
            return
        if goal_state in ("stalled", "invalid", "timeout"):
            self._fail_frontier(now_s, prior_subgoal, "subgoal_" + goal_state)
        root = self._nearest_cell(self.vehicle)
        if root is None and now_s >= self.next_frontier_search_s:
            radius_cells = int(math.ceil(1.5 / self.grid_m))
            root = self._connected_recovery_anchor(
                self.vehicle, self._attachment_candidates(self.vehicle, radius_cells), (), now_s
            )
        if root is None:
            if now_s < self.next_frontier_search_s:
                return
            self.next_frontier_search_s = now_s + self.frontier_search_period_s
            self._prune_memories(now_s)
            self._clear_global_goal()
            bootstrap = self._bootstrap_waypoint()
            if bootstrap is not None:
                self.global_goal_kind = "bootstrap"
                self._publish_subgoal(
                    bootstrap, bootstrap, 0, now_s, row, "bootstrap_lidar_blind_zone"
                )
            else:
                row["closure_state"] = "INSUFFICIENT_REACHABILITY"
                self._recover_reachability((), now_s, row)
            row["decision_ms"] = 1000.0 * (time.perf_counter() - started)
            self._trace(row)
            return
        if not self.recent_roots or root != self.recent_roots[-1]:
            self.recent_roots.append(root)

        # Keep the selected frontier across multiple local waypoints. Reaching
        # a short controller subgoal must not choose a new frontier in a
        # potentially opposite direction.
        if self.global_goal is not None:
            global_distance = math.hypot(
                self.global_goal[0] * self.grid_m - self.vehicle[0],
                self.global_goal[1] * self.grid_m - self.vehicle[1],
            )
            global_height = self._key_to_pose(self.global_goal)[2]
            arrival_radius = self._global_arrival_radius()
            if (global_distance <= arrival_radius
                    and abs(global_height - self.vehicle[2]) <= self.goal_z_tolerance_m):
                if self.global_observe_started_s is None:
                    self.global_observe_started_s = now_s
                    self.global_observe_revision = self.space.revision
                    self._log_info(
                        "FLEX-E safe %s viewpoint reached; observing physical region",
                        self.global_goal_kind,
                    )
                observe_elapsed = now_s - self.global_observe_started_s
                scan_arrived = self.space.revision > self.global_observe_revision
                if (
                    observe_elapsed < self.frontier_observation_wait_s
                    or (
                        not scan_arrived
                        and observe_elapsed < 2.0 * self.frontier_observation_wait_s
                    )
                ):
                    row["reason"] = "observing_%s_region" % self.global_goal_kind
                else:
                    row["reason"] = self._finish_frontier_observation(now_s)
            else:
                parents, costs, expanded = self._path_to_goal(
                    root,
                    self.global_goal,
                    now_s,
                    self.global_goal_search_range_m,
                    self.global_goal_max_expansions,
                )
                row["expanded"] = expanded
                if parents is not None:
                    source_region = self._region_for_cell(root)
                    subgoal = self._egress_prefix(
                        root,
                        self.global_goal,
                        parents,
                        costs,
                        source_region,
                        (
                            self.minimum_frontier_waypoint_clearance_m
                            if self.global_goal_kind in ("frontier", "portal")
                            else self.minimum_waypoint_clearance_m
                        ),
                    )
                    self._publish_subgoal(
                        subgoal,
                        self.global_goal,
                        self.global_goal_gain,
                        now_s,
                        row,
                        "continued_global_%s" % self.global_goal_kind,
                    )
                else:
                    self._log_warning("FLEX-E global region goal is no longer graph-reachable")
                    self._fail_frontier(now_s, None, "region_goal_unreachable")

        if self.global_goal is None and self.active_goal is None:
            if now_s < self.next_frontier_search_s:
                return
            searches = []
            if self.recovery_started_s is None:
                searches.extend([
                    ("local", {"frontier_only": True}),
                    ("global_fallback", {
                        "search_range_m": self.global_search_range_m,
                        "max_expansions": self.global_max_expansions,
                        "frontier_only": True,
                    }),
                ])
            searches.append(("closure_audit", {
                "search_range_m": self.completion_search_range_m,
                "max_expansions": self.completion_max_expansions,
                "min_goal_distance_m": self.cleanup_goal_distance_m,
                "allow_root_viewpoint": True,
                "resolve_orphans": True,
            }))
            for search_mode, options in searches:
                (
                    target,
                    viewpoint,
                    gain,
                    information_gain,
                    directions,
                    expanded,
                    parents,
                    costs,
                    frontier_stats,
                    selected_region,
                    target_kind,
                ) = self._select_frontier(
                    root, self.vehicle, now_s, **options
                )
                if target is not None:
                    break
            root = frontier_stats["root"]
            if frontier_stats["valid_attachment"] and self.recovery_started_s is not None:
                self._log_info(
                    "FLEX-E graph attachment restored: root=%s reachable=%d",
                    root, frontier_stats["reachable"],
                )
                self.recovery_started_s = None
                self.recovery_attempts = 0
            row.update(
                {
                    "expanded": expanded,
                    "frontiers": frontier_stats["raw"],
                    "frontier_regions": frontier_stats["regions"],
                    "selectable_frontiers": frontier_stats["selectable_frontiers"],
                    "selectable_coverage": frontier_stats["selectable_coverage"],
                    "uncovered_support_cells": frontier_stats["uncovered_cells"],
                    "connected_gap_cells": frontier_stats["gap_cells"],
                    "unresolved_regions": frontier_stats["unresolved"],
                    "coverage_ratio": frontier_stats["selected_coverage_ratio"],
                    "exploring_regions": frontier_stats["exploring_regions"],
                    "covered_regions": frontier_stats["covered_regions"],
                    "deferred_regions": frontier_stats["deferred_regions"],
                    "inaccessible_regions": frontier_stats["inaccessible_regions"],
                    "orphaned_regions": frontier_stats["orphaned"],
                    "global_region_route": frontier_stats["route_regions"],
                    "selected_region": "" if selected_region is None else str(selected_region),
                    "target_kind": target_kind,
                    "reachable_cells": frontier_stats["reachable"],
                    "audit_root": str(root),
                    "minimum_audit_reachable": frontier_stats["minimum_reachable"],
                    "recovery_attempts": self.recovery_attempts,
                    "search_truncated": frontier_stats["truncated"],
                    "information_gain": information_gain,
                }
            )

            if target is None:
                last_growth_s = max(
                    value
                    for value in (self.last_graph_growth_s, self.last_space_growth_s, 0.0)
                    if value is not None
                )
                map_quiet_s = max(0.0, now_s - last_growth_s)
                closure_state = self.space.closure_state(
                    frontier_stats["unresolved"],
                    bool(frontier_stats["truncated"]),
                    frontier_stats["reachable"],
                    frontier_stats["minimum_reachable"],
                    map_quiet_s,
                    self.completion_map_quiet_s,
                )
                if not frontier_stats["valid_attachment"] and not frontier_stats["truncated"]:
                    closure_state = "INSUFFICIENT_REACHABILITY"
                row["closure_state"] = closure_state
                if (
                    closure_state == "INSUFFICIENT_REACHABILITY"
                    or self.recovery_started_s is not None
                ):
                    self._recover_reachability(costs, now_s, row)
                elif closure_state == "CLOSED":
                    self.coverage_stall_started_s = None
                    if self.no_frontier_started_s is None:
                        self.no_frontier_started_s = now_s
                    no_frontier_s = now_s - self.no_frontier_started_s
                    self.completion_audit_count += 1
                    self.next_frontier_search_s = now_s + self.global_audit_period_s
                    if (
                        elapsed_s >= self.minimum_exploration_s
                        and no_frontier_s >= self.no_frontier_finish_s
                        and self.completion_audit_count >= self.completion_stable_audits
                    ):
                        self._finish_exploration("reachable_space_closed", now_s, row)
                    else:
                        row["reason"] = "confirming_reachable_space_closure"
                else:
                    self.no_frontier_started_s = None
                    self.completion_audit_count = 0
                    self.next_frontier_search_s = now_s + self.frontier_search_period_s
                    row["reason"] = "closure_audit_" + closure_state.lower()
                    if closure_state == "OPEN":
                        if (
                            self.coverage_stall_started_s is None
                            or self.coverage_stall_revision != self.space.coverage_revision
                        ):
                            self.coverage_stall_started_s = now_s
                            self.coverage_stall_revision = self.space.coverage_revision
                        if now_s - self.coverage_stall_started_s >= self.coverage_stall_timeout_s:
                            self._finish_exploration("coverage_stalled", now_s, row)
                    else:
                        self.coverage_stall_started_s = None
                self._log_info_throttle(
                    5.0,
                    "FLEX-E closure audit: state=%s raw=%d frontier_regions=%d uncovered_support=%d connected_gap=%d unresolved=%d exploring=%d deferred=%d inaccessible=%d orphaned=%d selectable=%d reachable=%d expanded=%d truncated=%d quiet=%.1fs stable=%d/%d",
                    closure_state,
                    frontier_stats["raw"],
                    frontier_stats["regions"],
                    frontier_stats["uncovered_cells"],
                    frontier_stats["gap_cells"],
                    frontier_stats["unresolved"],
                    frontier_stats["exploring_regions"],
                    frontier_stats["deferred_regions"],
                    frontier_stats["inaccessible_regions"],
                    frontier_stats["orphaned"],
                    frontier_stats["selectable"],
                    frontier_stats["reachable"],
                    expanded,
                    frontier_stats["truncated"],
                    map_quiet_s,
                    self.completion_audit_count,
                    self.completion_stable_audits,
                )
                row["decision_ms"] = 1000.0 * (time.perf_counter() - started)
                self._last_runtime_s = row["decision_ms"] / 1000.0
                self._trace(row)
                if self.finished:
                    self._timer_stop_requested = True
                return
            self.no_frontier_started_s = None
            self.coverage_stall_started_s = None
            self.completion_audit_count = 0
            self._closure_closed = False
            self.next_frontier_search_s = 0.0
            self.global_goal, self.frontier_anchor, self.global_goal_gain = viewpoint, target, gain
            self.active_region = selected_region
            self.global_goal_kind = target_kind
            self.global_coverage_keys = tuple(
                frontier_stats["selected_coverage_keys"]
            )
            self.global_region_support_keys = tuple(
                frontier_stats["selected_region_support_keys"]
            )
            self.global_coverage_before = sum(
                self.space.is_support_key_covered(key)
                for key in self.global_coverage_keys
            )
            self.global_coverage_revision = self.space.coverage_revision
            self.frontier_anchor_directions = directions
            self.frontier_unknown_before = information_gain
            self.global_goal_selected_s = now_s
            self.global_observe_started_s = None
            self.global_observe_revision = self.space.revision
            if search_mode == "global_fallback":
                self.global_goal_search_range_m = self.global_search_range_m
                self.global_goal_max_expansions = self.global_max_expansions
            elif search_mode == "closure_audit":
                self.global_goal_search_range_m = frontier_stats["search_range_m"]
                self.global_goal_max_expansions = self.completion_max_expansions
            else:
                self.global_goal_search_range_m = self.search_range_m
                self.global_goal_max_expansions = self.max_expansions
            self.last_goal = target
            source_region = self._region_for_cell(root)
            subgoal = self._egress_prefix(
                root,
                viewpoint,
                parents,
                costs,
                source_region,
                (
                    self.minimum_frontier_waypoint_clearance_m
                    if target_kind in ("frontier", "portal")
                    else self.minimum_waypoint_clearance_m
                ),
            )
            if target_kind == "coverage":
                reason = "new_%s_interior_coverage_information_%d" % (
                    search_mode,
                    information_gain,
                )
            elif target_kind == "portal":
                reason = "new_%s_portal_information_%d" % (search_mode, information_gain)
            else:
                reason = (
                    "new_%s_true_frontier_direction_%d_information_%d"
                    % (search_mode, gain, information_gain)
                )
            self._publish_subgoal(
                subgoal,
                viewpoint,
                gain,
                now_s,
                row,
                reason,
            )

        row["decision_ms"] = 1000.0 * (time.perf_counter() - started)
        self._last_runtime_s = row["decision_ms"] / 1000.0
        self._trace(row)
