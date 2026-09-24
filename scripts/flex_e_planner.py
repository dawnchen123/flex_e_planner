#!/usr/bin/env python3
"""FLEX-E online multi-level frontier planner for AEDE.

The planner receives AEDE's public terrain, registered lidar and truth-pose
topics. It never loads the Garage mesh or an offline point cloud. Information
targets lie on real free/unknown boundaries; every executable target remains
an observed, support-connected surface cell published on ``/way_point``.
"""

from __future__ import print_function

import csv
import heapq
import math
import os
import random
import threading
import time
from collections import defaultdict

import message_filters
import numpy as np
import rospy
from flex_e_core.space_evaluator import ExplorationSpaceEvaluator
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool, Float32, Header


def _pointcloud_xyzi(message):
    """Decode an AEDE PointCloud2 without assuming field order."""

    fields = {field.name: field for field in message.fields}
    names = ["x", "y", "z"]
    if any(name not in fields for name in names):
        raise ValueError("point cloud has no x/y/z fields")
    if "intensity" in fields:
        names.append("intensity")
    type_codes = {
        PointField.INT8: "i1",
        PointField.UINT8: "u1",
        PointField.INT16: "i2",
        PointField.UINT16: "u2",
        PointField.INT32: "i4",
        PointField.UINT32: "u4",
        PointField.FLOAT32: "f4",
        PointField.FLOAT64: "f8",
    }
    endian = ">" if message.is_bigendian else "<"
    formats, offsets = [], []
    for name in names:
        field = fields[name]
        if field.datatype not in type_codes or field.count != 1:
            raise ValueError("unsupported point-cloud field %s" % name)
        formats.append(endian + type_codes[field.datatype])
        offsets.append(field.offset)
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": message.point_step})
    points = np.frombuffer(message.data, dtype=dtype, count=message.width * message.height)
    xyz = np.column_stack((points["x"], points["y"], points["z"])).astype(np.float64, copy=False)
    intensity = np.asarray(points["intensity"], dtype=np.float64) if "intensity" in names else np.zeros(len(points))
    return xyz, intensity


class FlexEPlanner(object):
    """Layer-aware Dijkstra frontier selection over observed terrain support."""

    def __init__(self):
        param = lambda name, default: rospy.get_param("~" + name, default)
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
        self.minimum_exploration_s = max(0.0, float(param("minimum_exploration_s", 120.0)))
        self.max_exploration_s = max(0.0, float(param("max_exploration_s", 3600.0)))
        self.shutdown_on_finish = bool(param("shutdown_on_finish", False))
        self.finish_shutdown_delay_s = max(0.1, float(param("finish_shutdown_delay_s", 2.0)))
        self.trace_idle_period_s = max(self.period_s, float(param("trace_idle_period_s", 2.0)))
        self.rng = random.Random(int(param("seed", 0)))
        self.trace_path = str(param("decision_trace_path", "")).strip()

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
            max_voxels=param("occupancy_max_voxels", 400000),
            completed_resolution_m=param("completed_region_resolution_m", 1.0),
            completed_vertical_tolerance_m=param(
                "completed_region_vertical_tolerance_m", 0.75
            ),
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
        self.active_goal = None
        self.active_goal_reached_latched = False
        self.active_started_s = None
        self.active_last_progress_s = None
        self.active_best_distance_m = None
        self.global_goal = None
        self.frontier_anchor = None
        self.frontier_anchor_directions = ()
        self.frontier_unknown_before = 0
        self.global_goal_gain = 0
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

        self.waypoint_pub = rospy.Publisher(self.waypoint_topic, PointStamped, queue_size=5)
        self.runtime_pub = rospy.Publisher(self.runtime_topic, Float32, queue_size=20)
        self.finish_pub = rospy.Publisher(self.finish_topic, Bool, queue_size=1, latch=True)
        self.stop_pub = rospy.Publisher(self.stop_topic, Bool, queue_size=1, latch=True)
        self.closed_pub = rospy.Publisher("/flex_e/exploration_closed", Bool, queue_size=1, latch=True)
        self.frontier_debug_pub = rospy.Publisher(
            "/flex_e/true_frontiers", PointCloud2, queue_size=1
        )
        self.viewpoint_debug_pub = rospy.Publisher(
            "/flex_e/safe_viewpoints", PointCloud2, queue_size=1
        )
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._odom_callback, queue_size=50)
        self.terrain_sub = rospy.Subscriber(self.terrain_topic, PointCloud2, self._terrain_callback, queue_size=2)
        self.scan_sub = message_filters.Subscriber(self.registered_scan_topic, PointCloud2)
        self.scan_odom_sub = message_filters.Subscriber(self.scan_odom_topic, Odometry)
        self.scan_sync = message_filters.ApproximateTimeSynchronizer(
            [self.scan_sub, self.scan_odom_sub], queue_size=20, slop=0.05
        )
        self.scan_sync.registerCallback(self._registered_scan_callback)
        self.timer = rospy.Timer(rospy.Duration(self.period_s), self._plan)
        rospy.on_shutdown(self._close_trace)
        rospy.loginfo(
            "FLEX-E ready: terrain=%s scan=%s scan_pose=%s odometry=%s waypoint=%s grid=%.2f/%.2f m",
            self.terrain_topic,
            self.registered_scan_topic,
            self.scan_odom_topic,
            self.odom_topic,
            self.waypoint_topic,
            self.grid_m,
            self.z_grid_m,
        )

    def _open_trace(self):
        if not self.trace_path:
            return
        parent = os.path.dirname(os.path.abspath(self.trace_path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        fields = (
            "sim_time_s",
            "decision_ms",
            "cell_count",
            "occupancy_voxels",
            "completed_region_cells",
            "expanded",
            "frontiers",
            "frontier_regions",
            "reachable_cells",
            "search_truncated",
            "closure_state",
            "information_gain",
            "target_x",
            "target_y",
            "target_z",
            "reason",
        )
        self.trace_file = open(self.trace_path, "a+", newline="", encoding="utf-8")
        self.trace_file.seek(0)
        existing_header = self.trace_file.readline().strip()
        if existing_header:
            fields = tuple(next(csv.reader([existing_header])))
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

    def _odom_callback(self, message):
        with self.state_lock:
            self._odom_callback_locked(message)

    def _odom_callback_locked(self, message):
        point = message.pose.pose.position
        orientation = message.pose.pose.orientation
        stamp = message.header.stamp.to_sec()
        previous = self.vehicle
        self.vehicle = (float(point.x), float(point.y), float(point.z), stamp)
        sin_yaw = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cos_yaw = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        self.vehicle_yaw = math.atan2(sin_yaw, cos_yaw)

        # The controller can cross a short subgoal between two planner timer
        # callbacks. Latch the crossing here at odometry rate so a passed
        # waypoint is never commanded behind the vehicle on the next cycle.
        if self.active_goal is not None:
            goal_x = self.active_goal[0] * self.grid_m
            goal_y = self.active_goal[1] * self.grid_m
            distance = math.hypot(goal_x - point.x, goal_y - point.y)
            crossed = distance <= self.goal_reached_m
            if not crossed and previous is not None:
                crossed = self._point_segment_distance(
                    goal_x, goal_y, previous[0], previous[1], float(point.x), float(point.y)
                ) <= self.goal_reached_m
            if crossed:
                self.active_goal_reached_latched = True
        if self.last_visit is None or math.hypot(point.x - self.last_visit[0], point.y - self.last_visit[1]) >= self.visit_sample_distance_m:
            ix = int(math.floor(float(point.x) / self.grid_m + 0.5))
            iy = int(math.floor(float(point.y) / self.grid_m + 0.5))
            ground_z = float(point.z) - self.vehicle_height_m
            iz = int(math.floor(ground_z / self.z_grid_m + 0.5))
            self.visited_index[(ix, iy)].add(iz)
            self.last_visit = (float(point.x), float(point.y), float(point.z))

    @staticmethod
    def _point_segment_distance(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq <= 1.0e-12:
            return math.hypot(px - ax, py - ay)
        ratio = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
        return math.hypot(px - (ax + ratio * dx), py - (ay + ratio * dy))

    def _terrain_callback(self, message):
        with self.state_lock:
            self._terrain_callback_locked(message)

    def _terrain_callback_locked(self, message):
        try:
            xyz, intensity = _pointcloud_xyzi(message)
        except ValueError as error:
            rospy.logwarn_throttle(5.0, "FLEX-E terrain decode: %s", error)
            return
        valid = np.isfinite(xyz).all(axis=1) & np.isfinite(intensity) & (intensity <= self.safe_intensity)
        xyz = xyz[valid]
        if not len(xyz):
            return
        ix = np.floor(xyz[:, 0] / self.grid_m + 0.5).astype(np.int64)
        iy = np.floor(xyz[:, 1] / self.grid_m + 0.5).astype(np.int64)
        iz = np.floor(xyz[:, 2] / self.z_grid_m + 0.5).astype(np.int64)
        keys = np.column_stack((ix, iy, iz))
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        z_sums = np.bincount(inverse, weights=xyz[:, 2])
        stamp = message.header.stamp.to_sec() or rospy.Time.now().to_sec()
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

    def _registered_scan_callback(self, scan, scan_odom):
        with self.state_lock:
            stamp = scan.header.stamp.to_sec() or rospy.Time.now().to_sec()
            if self.last_visibility_update_s is not None:
                delta_s = stamp - self.last_visibility_update_s
                if 0.0 <= delta_s < self.visibility_update_period_s:
                    return
            try:
                xyz, _intensity = _pointcloud_xyzi(scan)
            except ValueError as error:
                rospy.logwarn_throttle(5.0, "FLEX-E registered scan decode: %s", error)
                return
            point = scan_odom.pose.pose.position
            origin = (float(point.x), float(point.y), float(point.z))
            voxel_count_before = len(self.space.evidence)
            self.space.integrate_scan(origin, xyz)
            if (
                len(self.space.evidence) - voxel_count_before
                >= self.occupancy_growth_min_voxels
            ):
                self.last_space_growth_s = stamp
                self.completion_audit_count = 0
                self.no_frontier_started_s = None
            self.last_visibility_update_s = stamp

    def _trim_old_cells(self):
        for key in sorted(self.cells, key=lambda item: self.cells[item][2])[:max(1, self.max_cells // 20)]:
            self.cells.pop(key, None)
            self.cell_first_seen_s.pop(key, None)
            levels = self.levels.get(key[:2])
            if levels:
                levels.discard(key)
                if not levels:
                    self.levels.pop(key[:2], None)

    def _nearest_cell(self, vehicle):
        ix, iy = (int(math.floor(value / self.grid_m + 0.5)) for value in vehicle[:2])
        best, score = None, float("inf")
        for radius in range(7):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for key in self.levels.get((ix + dx, iy + dy), ()):
                        dz = self.cells[key][0] + self.vehicle_height_m - vehicle[2]
                        candidate = (dx * self.grid_m) ** 2 + (dy * self.grid_m) ** 2 + 0.2 * dz * dz
                        if candidate < score:
                            best, score = key, candidate
            if best is not None:
                return best
        return min(self.cells, key=lambda key: (key[0] * self.grid_m - vehicle[0]) ** 2 + (key[1] * self.grid_m - vehicle[1]) ** 2)

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
                    if self._edge_ok(key, candidate):
                        yield candidate

    def _frontier_evidence(self, key, include_completed=False):
        """Return visible unknown directions, information gain and risk."""

        directions = []
        unknown_columns = 0
        occupied_columns = 0
        elevation = self.cells[key][0]
        tolerance = self.max_step_m + self.grid_m * math.tan(self.max_slope)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            candidates = self.levels.get((key[0] + dx, key[1] + dy), ())
            if any(abs(self.cells[item][0] - elevation) <= tolerance for item in candidates):
                continue
            x, y = key[0] * self.grid_m, key[1] * self.grid_m
            if not include_completed and self.space.is_completed(
                x, y, elevation, dx, dy
            ):
                continue
            evidence = self.space.direction_evidence(
                x, y, elevation, dx, dy, self.grid_m
            )
            if not evidence.is_frontier:
                continue
            directions.append((dx, dy))
            unknown_columns += evidence.unknown_columns
            occupied_columns += evidence.occupied_columns
        return tuple(directions), unknown_columns, occupied_columns

    def _frontier_gain(self, key):
        directions, _unknown_columns, _occupied_columns = self._frontier_evidence(key)
        return len(directions)

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

    def _prune_memories(self, now_s):
        self.blocked_subgoals = [
            item for item in self.blocked_subgoals if now_s - item[3] <= self.blocked_subgoal_memory_s
        ]

    def _clear_global_goal(self):
        self.active_goal = None
        self.active_goal_reached_latched = False
        self.global_goal = None
        self.frontier_anchor = None
        self.frontier_anchor_directions = ()
        self.frontier_unknown_before = 0
        self.global_goal_gain = 0
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
        rospy.loginfo(
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
        self._mark_frontier_region(self.completed_region_radius_m, reason)
        rospy.logwarn("FLEX-E physical frontier region rejected: %s", reason)
        self._clear_global_goal()

    def _finish_frontier_observation(self):
        if self.frontier_anchor is None:
            self._clear_global_goal()
            return "frontier_observed_without_anchor"
        directions, unknown_after, _risk = self._frontier_evidence(
            self.frontier_anchor, include_completed=True
        )
        growth = self._new_cell_count(
            self.frontier_anchor, self.global_goal_selected_s
        )
        if not directions or unknown_after < max(1, self.frontier_unknown_before // 3):
            outcome = "frontier_region_closed"
            radius_m = self.completed_region_radius_m
        elif growth >= self.frontier_min_growth_cells:
            outcome = "frontier_region_advanced"
            radius_m = self.completed_region_progress_radius_m
        else:
            outcome = "frontier_region_no_information_gain"
            radius_m = self.completed_region_radius_m
        self._mark_frontier_region(radius_m, outcome)
        rospy.loginfo(
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
                if candidate < costs.get(neighbor, float("inf")) and candidate <= search_range_m:
                    costs[neighbor], parents[neighbor] = candidate, key
                    heapq.heappush(queue, (candidate, neighbor))
        truncated = expanded >= max_expansions and any(
            cost == costs.get(key) and cost <= search_range_m for cost, key in queue
        )
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

    def _publish_frontier_debug(self, items, viewpoints):
        if not self.publish_debug_clouds:
            return
        header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id)
        frontier_points = [
            (item[1][0] * self.grid_m, item[1][1] * self.grid_m, self.cells[item[1]][0])
            for item in items
        ]
        viewpoint_points = [
            (key[0] * self.grid_m, key[1] * self.grid_m, self.cells[key][0])
            for key in viewpoints
        ]
        self.frontier_debug_pub.publish(
            point_cloud2.create_cloud_xyz32(header, frontier_points)
        )
        self.viewpoint_debug_pub.publish(
            point_cloud2.create_cloud_xyz32(header, viewpoint_points)
        )

    def _select_frontier(
        self,
        root,
        vehicle,
        now_s,
        search_range_m=None,
        max_expansions=None,
        min_goal_distance_m=None,
        allow_root_viewpoint=False,
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
        items = []
        for key, cost in costs.items():
            if key == root or cost > search_range_m:
                continue
            directions, information_gain, risk = self._frontier_evidence(key)
            gain = len(directions)
            neighbors = list(self._neighbors(key))
            if not (0 < gain <= self.max_frontier_gain and len(neighbors) >= self.min_frontier_support):
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
            items.append(
                (score, key, gain, information_gain, risk, cost, direct, directions)
            )

        clusters = self._cluster_frontiers(items)
        choices, debug_viewpoints = [], []
        stats = {
            "raw": len(items),
            "regions": len(clusters),
            "selectable": 0,
            "unresolved": len(clusters),
            "reachable": len(costs),
            "truncated": int(truncated),
        }
        for cluster in clusters:
            candidates = []
            for item in cluster:
                score, key, gain, information_gain, risk, cost, direct, directions = item
                if direct < min_goal_distance_m:
                    continue
                viewpoint = None
                for option in self._frontier_viewpoint_options(
                    root, key, parents, costs
                ):
                    if option == root and not allow_root_viewpoint:
                        continue
                    if self._blocked(option):
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
                    )
                )
            if not candidates:
                continue
            best = max(candidates)
            choices.append(
                (best[0] + self.frontier_cluster_gain_reward * math.sqrt(len(cluster)), best)
            )
            debug_viewpoints.append(best[2])

        self._publish_frontier_debug(items, debug_viewpoints)
        stats["selectable"] = len(choices)
        if choices:
            _region_score, best = max(choices, key=lambda item: item[0])
            _score, key, viewpoint, gain, information_gain, _risk, _cost, directions = best
            return (
                key,
                viewpoint,
                gain,
                information_gain,
                directions,
                expanded,
                parents,
                costs,
                stats,
            )
        return None, None, 0, 0, (), expanded, parents, costs, stats

    def _path_to_goal(self, root, target, now_s, search_range_m, max_expansions):
        if target not in self.cells:
            return None, None, 0
        self._prune_memories(now_s)
        queue, costs, parents = [(0.0, root)], {root: 0.0}, {root: None}
        expanded = 0
        while queue and expanded < max_expansions:
            cost, key = heapq.heappop(queue)
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
                    heapq.heappush(queue, (candidate, neighbor))
        return None, None, expanded

    def _route_prefix(self, root, target, parents, costs):
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
            if not self._blocked(key):
                choice = key
        if choice != root or len(chain) == 1:
            return choice
        for key in reversed(chain[:-1]):
            if not self._blocked(key):
                return key
        return root

    def _frontier_viewpoint_options(self, root, target, parents, costs):
        """Order observed support cells by proximity to the desired standoff."""

        chain, cursor = [], target
        while cursor is not None:
            chain.append(cursor)
            cursor = parents.get(cursor)
        if not chain or chain[-1] != root:
            return (target,)
        target_cost = costs[target]
        return tuple(
            sorted(
                chain,
                key=lambda key: (
                    abs((target_cost - costs.get(key, 0.0)) - self.frontier_standoff_m),
                    -costs.get(key, 0.0),
                ),
            )
        )

    def _active_goal_state(self, now_s):
        if self.active_goal is None or self.vehicle is None:
            return "none"
        if self.active_goal not in self.cells:
            self.active_goal = None
            self.active_goal_reached_latched = False
            return "invalid"
        x, y = self.active_goal[0] * self.grid_m, self.active_goal[1] * self.grid_m
        distance = math.hypot(x - self.vehicle[0], y - self.vehicle[1])
        if self.active_goal_reached_latched or distance <= self.goal_reached_m:
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
        waypoint = PointStamped()
        waypoint.header.frame_id, waypoint.header.stamp = self.frame_id, rospy.Time.now()
        waypoint.point.x, waypoint.point.y = target[0] * self.grid_m, target[1] * self.grid_m
        waypoint.point.z = self.cells[target][0] + self.vehicle_height_m
        self.waypoint_pub.publish(waypoint)
        self.active_goal = target
        self.active_goal_reached_latched = False
        self.active_started_s = self.active_last_progress_s = now_s
        self.active_best_distance_m = math.hypot(waypoint.point.x - self.vehicle[0], waypoint.point.y - self.vehicle[1])
        row.update(
            {
                "target_x": waypoint.point.x,
                "target_y": waypoint.point.y,
                "target_z": waypoint.point.z,
                "reason": reason,
            }
        )
        rospy.loginfo(
            "FLEX-E %s subgoal=(%.2f, %.2f) global=(%.2f, %.2f) gain=%d",
            reason,
            waypoint.point.x,
            waypoint.point.y,
            final_goal[0] * self.grid_m,
            final_goal[1] * self.grid_m,
            gain,
        )

    def _finish_exploration(self, reason, now_s, row):
        if self.finished:
            return
        self.finished = True
        self._clear_global_goal()
        if self.vehicle is not None:
            stop = PointStamped()
            stop.header.frame_id, stop.header.stamp = self.frame_id, rospy.Time.now()
            stop.point.x, stop.point.y, stop.point.z = self.vehicle[:3]
            self.waypoint_pub.publish(stop)
            row.update({"target_x": stop.point.x, "target_y": stop.point.y, "target_z": stop.point.z})
        self.finish_pub.publish(Bool(data=True))
        self.stop_pub.publish(Bool(data=True))
        self.closed_pub.publish(Bool(data=reason == "reachable_space_closed"))
        row["reason"] = "exploration_finished_" + reason
        rospy.loginfo("FLEX-E exploration finished: %s at %.1f s", reason, now_s)
        if self.shutdown_on_finish:
            self.shutdown_timer = rospy.Timer(
                rospy.Duration(self.finish_shutdown_delay_s),
                lambda _event: rospy.signal_shutdown("exploration finished: " + reason),
                oneshot=True,
            )

    def _plan(self, _event):
        if not self.state_lock.acquire(False):
            return
        try:
            self._plan_locked(_event)
        finally:
            self.state_lock.release()

    def _plan_locked(self, _event):
        started, now_s = time.perf_counter(), rospy.Time.now().to_sec()
        row = {
            "sim_time_s": now_s,
            "cell_count": len(self.cells),
            "occupancy_voxels": len(self.space.evidence),
            "completed_region_cells": len(self.space.completed_regions),
            "expanded": 0,
            "frontiers": 0,
            "frontier_regions": 0,
            "reachable_cells": 0,
            "search_truncated": 0,
            "closure_state": "",
            "information_gain": 0,
            "target_x": "",
            "target_y": "",
            "target_z": "",
            "reason": "",
        }
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
            self.runtime_pub.publish(Float32(data=row["decision_ms"] / 1000.0))
            self._trace(row)
            self.timer.shutdown()
            return
        prior_subgoal = self.active_goal
        goal_state = self._active_goal_state(now_s)
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

        # Keep the selected frontier across multiple local waypoints. Reaching
        # a short controller subgoal must not choose a new frontier in a
        # potentially opposite direction.
        if self.global_goal is not None:
            global_distance = math.hypot(
                self.global_goal[0] * self.grid_m - self.vehicle[0],
                self.global_goal[1] * self.grid_m - self.vehicle[1],
            )
            if global_distance <= self.global_goal_reached_m:
                if self.global_observe_started_s is None:
                    self.global_observe_started_s = now_s
                    self.global_observe_revision = self.space.revision
                    rospy.loginfo(
                        "FLEX-E safe frontier viewpoint reached; observing physical region"
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
                    row["reason"] = "observing_frontier_region"
                else:
                    row["reason"] = self._finish_frontier_observation()
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
                    subgoal = self._route_prefix(root, self.global_goal, parents, costs)
                    self._publish_subgoal(
                        subgoal,
                        self.global_goal,
                        self.global_goal_gain,
                        now_s,
                        row,
                        "continued_global_frontier",
                    )
                else:
                    rospy.logwarn("FLEX-E global frontier is no longer graph-reachable")
                    self._fail_frontier(now_s, None, "frontier_unreachable")

        if self.global_goal is None and self.active_goal is None:
            if now_s < self.next_frontier_search_s:
                return
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
            ) = self._select_frontier(root, self.vehicle, now_s)
            search_mode = "local"
            if target is None:
                rospy.loginfo_throttle(
                    5.0,
                    "FLEX-E local frontier search exhausted; expanding to %.1f m / %d nodes",
                    self.global_search_range_m,
                    self.global_max_expansions,
                )
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
                ) = self._select_frontier(
                    root,
                    self.vehicle,
                    now_s,
                    self.global_search_range_m,
                    self.global_max_expansions,
                )
                search_mode = "global_fallback"
            if target is None:
                rospy.loginfo_throttle(
                    5.0,
                    "FLEX-E global frontier search exhausted; auditing reachable closure",
                )
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
                ) = self._select_frontier(
                    root,
                    self.vehicle,
                    now_s,
                    self.completion_search_range_m,
                    self.completion_max_expansions,
                    min_goal_distance_m=self.cleanup_goal_distance_m,
                    allow_root_viewpoint=True,
                )
                search_mode = "closure_audit"
            row.update(
                {
                    "expanded": expanded,
                    "frontiers": frontier_stats["raw"],
                    "frontier_regions": frontier_stats["regions"],
                    "reachable_cells": frontier_stats["reachable"],
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
                    self.completion_min_reachable_cells,
                    map_quiet_s,
                    self.completion_map_quiet_s,
                )
                row["closure_state"] = closure_state
                self.closed_pub.publish(Bool(data=False))
                if closure_state == "CLOSED":
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
                rospy.loginfo_throttle(
                    5.0,
                    "FLEX-E closure audit: state=%s raw=%d regions=%d selectable=%d reachable=%d expanded=%d truncated=%d quiet=%.1fs stable=%d/%d",
                    closure_state,
                    frontier_stats["raw"],
                    frontier_stats["regions"],
                    frontier_stats["selectable"],
                    frontier_stats["reachable"],
                    expanded,
                    frontier_stats["truncated"],
                    map_quiet_s,
                    self.completion_audit_count,
                    self.completion_stable_audits,
                )
                row["decision_ms"] = 1000.0 * (time.perf_counter() - started)
                self.runtime_pub.publish(Float32(data=row["decision_ms"] / 1000.0))
                self._trace(row)
                if self.finished:
                    self.timer.shutdown()
                return
            self.no_frontier_started_s = None
            self.completion_audit_count = 0
            self.closed_pub.publish(Bool(data=False))
            self.next_frontier_search_s = 0.0
            self.global_goal, self.frontier_anchor, self.global_goal_gain = viewpoint, target, gain
            self.frontier_anchor_directions = directions
            self.frontier_unknown_before = information_gain
            self.global_goal_selected_s = now_s
            self.global_observe_started_s = None
            self.global_observe_revision = self.space.revision
            if search_mode == "global_fallback":
                self.global_goal_search_range_m = self.global_search_range_m
                self.global_goal_max_expansions = self.global_max_expansions
            elif search_mode == "closure_audit":
                self.global_goal_search_range_m = self.completion_search_range_m
                self.global_goal_max_expansions = self.completion_max_expansions
            else:
                self.global_goal_search_range_m = self.search_range_m
                self.global_goal_max_expansions = self.max_expansions
            self.last_goal = target
            subgoal = self._route_prefix(root, viewpoint, parents, costs)
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
        self.runtime_pub.publish(Float32(data=row["decision_ms"] / 1000.0))
        self._trace(row)


def main():
    rospy.init_node("flex_e_planner")
    FlexEPlanner()
    rospy.spin()


if __name__ == "__main__":
    main()
