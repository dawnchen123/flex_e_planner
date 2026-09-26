"""Sparse visibility, occupancy and reachable-space closure evaluation.

This module is deliberately independent of ROS.  The planner supplies a
truth-pose sensor origin and points that are already registered in the map
frame.  Ray interiors provide free-space evidence and returns provide
occupied-space evidence.  Terrain support remains the planner's source of
traversability; occupancy is used to decide whether a missing terrain neighbor
is a real free/unknown frontier or an already observed wall/drop boundary.
"""

from __future__ import division

import math

import numpy as np


UNKNOWN = 0
FREE = 1
OCCUPIED = 2


class DirectionEvidence(object):
    """Visibility result for one outward direction of a terrain cell."""

    __slots__ = ("is_frontier", "unknown_columns", "occupied_columns")

    def __init__(self, is_frontier=False, unknown_columns=0, occupied_columns=0):
        self.is_frontier = bool(is_frontier)
        self.unknown_columns = int(unknown_columns)
        self.occupied_columns = int(occupied_columns)


class ExplorationSpaceEvaluator(object):
    """Bounded sparse 3-D evidence map plus physical completion masks."""

    def __init__(
        self,
        resolution_m=0.5,
        ray_step_m=0.5,
        min_range_m=0.75,
        max_range_m=15.0,
        max_rays=900,
        occupied_threshold=2,
        free_threshold=-1,
        obstacle_inflation_m=0.6,
        min_clearance_m=0.3,
        max_clearance_m=1.6,
        frontier_lookahead_m=4.0,
        frontier_lateral_m=1.0,
        frontier_min_unknown_columns=3,
        max_voxels=400000,
        completed_resolution_m=1.0,
        completed_vertical_tolerance_m=0.75,
        support_coverage_resolution_m=0.5,
        support_coverage_range_m=12.0,
        support_coverage_max_cells=200000,
        support_coverage_min_known_fraction=0.65,
        clearance_search_m=3.0,
    ):
        self.resolution_m = max(0.1, float(resolution_m))
        self.ray_step_m = max(0.1, float(ray_step_m))
        self.min_range_m = max(0.0, float(min_range_m))
        self.max_range_m = max(self.min_range_m + self.resolution_m, float(max_range_m))
        self.max_rays = max(50, int(max_rays))
        self.occupied_threshold = max(1, int(occupied_threshold))
        self.free_threshold = min(-1, int(free_threshold))
        self.obstacle_inflation_m = max(0.0, float(obstacle_inflation_m))
        self.min_clearance_m = max(0.0, float(min_clearance_m))
        self.max_clearance_m = max(
            self.min_clearance_m + self.resolution_m, float(max_clearance_m)
        )
        self.frontier_lookahead_m = max(self.resolution_m, float(frontier_lookahead_m))
        self.frontier_lateral_m = max(0.0, float(frontier_lateral_m))
        self.frontier_min_unknown_columns = max(1, int(frontier_min_unknown_columns))
        self.max_voxels = max(1000, int(max_voxels))
        self.completed_resolution_m = max(
            self.resolution_m, float(completed_resolution_m)
        )
        self.completed_vertical_tolerance_m = max(
            self.resolution_m, float(completed_vertical_tolerance_m)
        )
        self.support_coverage_resolution_m = max(
            self.resolution_m, float(support_coverage_resolution_m)
        )
        self.support_coverage_range_m = max(
            self.support_coverage_resolution_m, float(support_coverage_range_m)
        )
        self.support_coverage_max_cells = max(1000, int(support_coverage_max_cells))
        self.support_coverage_min_known_fraction = min(
            1.0, max(0.0, float(support_coverage_min_known_fraction))
        )
        self.clearance_search_m = max(
            self.obstacle_inflation_m + self.resolution_m,
            float(clearance_search_m),
        )

        # Negative evidence is free; positive evidence is occupied.  Evidence
        # is bounded so a transient return can later be corrected by visibility.
        self.evidence = {}
        self.completed_regions = set()
        self.revision = 0
        self.completed_revision = 0
        # Coverage belongs to traversable support, not obstacle surfaces. A
        # support cell is persistent once a registered scan origin has a
        # robot-clear line of sight to it. Completed space can still be used
        # for transit, but it no longer attracts an information waypoint.
        self.covered_support = set()
        self.coverage_revision = 0
        self._column_cache = {}
        self._clearance_cache = {}

    @staticmethod
    def _round_cell(value, resolution):
        return int(math.floor(float(value) / resolution + 0.5))

    def integrate_scan(self, origin, points):
        """Ray-integrate a bounded, deterministic subset of a registered scan."""

        xyz = np.asarray(points, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("registered scan must be an N x 3 array")
        if not len(xyz):
            return 0
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        if not len(xyz):
            return 0
        offsets = xyz - origin
        ranges = np.linalg.norm(offsets, axis=1)
        valid = (ranges >= self.min_range_m) & (ranges <= self.max_range_m)
        xyz, ranges = xyz[valid], ranges[valid]
        if not len(xyz):
            return 0
        if len(xyz) > self.max_rays:
            indices = np.linspace(0, len(xyz) - 1, self.max_rays, dtype=np.int64)
            xyz, ranges = xyz[indices], ranges[indices]

        free_updates, occupied_updates = set(), set()
        resolution = self.resolution_m
        for point, distance in zip(xyz, ranges):
            endpoint = tuple(self._round_cell(value, resolution) for value in point)
            occupied_updates.add(endpoint)
            steps = max(
                0,
                int((float(distance) - 0.5 * resolution) / self.ray_step_m),
            )
            if not steps:
                continue
            direction = (point - origin) / float(distance)
            for step in range(1, steps + 1):
                position = origin + direction * (step * self.ray_step_m)
                key = tuple(self._round_cell(value, resolution) for value in position)
                if key != endpoint:
                    free_updates.add(key)

        # A return in this scan takes precedence over free rays crossing the
        # same coarse voxel.
        free_updates.difference_update(occupied_updates)
        for key in free_updates:
            self.evidence[key] = max(-5, self.evidence.get(key, 0) - 1)
        for key in occupied_updates:
            self.evidence[key] = min(5, self.evidence.get(key, 0) + 3)

        if len(self.evidence) > self.max_voxels:
            excess = len(self.evidence) - self.max_voxels
            for key in list(self.evidence)[:excess]:
                self.evidence.pop(key, None)
        if free_updates or occupied_updates:
            self.revision += 1
            self._column_cache.clear()
            self._clearance_cache.clear()
        return len(xyz)

    def _support_key(self, x, y, z):
        resolution = self.support_coverage_resolution_m
        return tuple(self._round_cell(value, resolution) for value in (x, y, z))

    def support_key(self, x, y, z):
        return self._support_key(x, y, z)

    def is_support_key_covered(self, key):
        return tuple(key) in self.covered_support

    def is_support_covered(self, x, y, z):
        return self._support_key(x, y, z) in self.covered_support

    def has_clear_support_line(
        self,
        sensor_position,
        support_position,
        sensor_height_m,
        minimum_known_fraction=0.0,
    ):
        """Return whether an inflated robot column can see a support cell.

        The ray follows the sensor-height profile above the terrain rather
        than tracing toward the ground return. Unknown columns are allowed:
        the registered scan is the observation that turns visible traversable
        support into covered space. Inflated occupied columns still occlude it.
        """

        sensor = np.asarray(sensor_position, dtype=np.float64).reshape(3)
        support = np.asarray(support_position, dtype=np.float64).reshape(3)
        target = support.copy()
        target[2] += float(sensor_height_m)
        distance = float(np.linalg.norm(target - sensor))
        if distance > self.support_coverage_range_m:
            return False
        if distance <= self.resolution_m:
            return True
        steps = max(1, int(math.ceil(distance / self.resolution_m)))
        known_columns = 0
        checked_columns = 0
        for step in range(1, steps):
            ratio = float(step) / steps
            sample = sensor + ratio * (target - sensor)
            ground_z = sample[2] - float(sensor_height_m)
            state = self.column_state(sample[0], sample[1], ground_z)
            checked_columns += 1
            if state == OCCUPIED:
                return False
            if state == FREE:
                known_columns += 1
        known_fraction = (
            float(known_columns) / checked_columns if checked_columns else 1.0
        )
        return known_fraction >= float(minimum_known_fraction)

    def observe_support_cells(self, sensor_position, support_points, sensor_height_m):
        """Persist support cells covered by one effective registered scan."""

        before = len(self.covered_support)
        for point in support_points:
            x, y, z = (float(value) for value in point)
            if self.column_state(x, y, z) == OCCUPIED:
                continue
            if self.has_clear_support_line(
                sensor_position,
                (x, y, z),
                sensor_height_m,
                self.support_coverage_min_known_fraction,
            ):
                self.covered_support.add(self._support_key(x, y, z))
        if len(self.covered_support) > self.support_coverage_max_cells:
            # Dict-like insertion order is unavailable for a set. Coverage is
            # deliberately bounded with a deterministic spatial ordering.
            keep = sorted(self.covered_support)[-self.support_coverage_max_cells :]
            self.covered_support = set(keep)
        added = len(self.covered_support) - before
        if added > 0:
            self.coverage_revision += 1
        return max(0, added)

    def support_visibility_gain(
        self,
        support_position,
        uncovered_points,
        sensor_height_m,
        max_checks=256,
        minimum_known_fraction=0.0,
    ):
        """Count uncovered traversable cells visible from a safe viewpoint."""

        return len(
            self.support_visibility_keys(
                support_position,
                uncovered_points,
                sensor_height_m,
                max_checks,
                minimum_known_fraction,
            )
        )

    def support_visibility_keys(
        self,
        support_position,
        uncovered_points,
        sensor_height_m,
        max_checks=256,
        minimum_known_fraction=0.0,
    ):
        """Return the exact support-key set used for predicted coverage gain."""

        points = list(uncovered_points)[: max(1, int(max_checks))]
        sensor = (
            float(support_position[0]),
            float(support_position[1]),
            float(support_position[2]) + float(sensor_height_m),
        )
        visible = []
        for point in points:
            key = self._support_key(*point)
            if key in self.covered_support:
                continue
            if self.has_clear_support_line(
                sensor,
                point,
                sensor_height_m,
                minimum_known_fraction,
            ):
                visible.append(key)
        return tuple(visible)

    def obstacle_clearance(self, x, y, ground_z, max_distance_m=None):
        """Distance from a support cell to the nearest observed obstacle."""

        max_distance_m = (
            self.clearance_search_m
            if max_distance_m is None
            else max(self.resolution_m, float(max_distance_m))
        )
        cache_key = (
            self._round_cell(x, self.resolution_m),
            self._round_cell(y, self.resolution_m),
            self._round_cell(ground_z, self.resolution_m),
            int(math.ceil(max_distance_m / self.resolution_m)),
        )
        cached = self._clearance_cache.get(cache_key)
        if cached is not None:
            return cached
        ix, iy, ground_iz, radius_cells = cache_key
        min_dz = max(1, int(math.ceil(self.min_clearance_m / self.resolution_m)))
        max_dz = max(min_dz, int(math.ceil(self.max_clearance_m / self.resolution_m)))
        best = max_distance_m
        for ox in range(-radius_cells, radius_cells + 1):
            for oy in range(-radius_cells, radius_cells + 1):
                distance = math.hypot(ox, oy) * self.resolution_m
                if distance >= best or distance > max_distance_m:
                    continue
                if any(
                    self.evidence.get((ix + ox, iy + oy, ground_iz + dz), 0)
                    >= self.occupied_threshold
                    for dz in range(min_dz, max_dz + 1)
                ):
                    best = max(0.0, distance - 0.5 * self.resolution_m)
        self._clearance_cache[cache_key] = best
        return best

    def column_state(self, x, y, ground_z, inflation_m=None):
        """Classify the robot-clearance column above a terrain support cell."""

        resolution = self.resolution_m
        inflation_m = (
            self.obstacle_inflation_m
            if inflation_m is None
            else max(0.0, float(inflation_m))
        )
        ix = self._round_cell(x, resolution)
        iy = self._round_cell(y, resolution)
        ground_iz = self._round_cell(ground_z, resolution)
        cache_key = (
            ix,
            iy,
            ground_iz,
            self._round_cell(inflation_m, 0.01),
        )
        cached = self._column_cache.get(cache_key)
        if cached is not None:
            return cached

        inflation_cells = int(math.ceil(inflation_m / resolution))
        # Never query the ground voxel itself: a valid floor return is occupied
        # in the ray map but must not be interpreted as a wall.
        min_dz = max(1, int(math.ceil(self.min_clearance_m / resolution)))
        max_dz = max(min_dz, int(math.ceil(self.max_clearance_m / resolution)))
        observed_free = False
        for iz in range(ground_iz + min_dz, ground_iz + max_dz + 1):
            if self.evidence.get((ix, iy, iz), 0) <= self.free_threshold:
                observed_free = True
            for ox in range(-inflation_cells, inflation_cells + 1):
                for oy in range(-inflation_cells, inflation_cells + 1):
                    # Occupancy keys represent finite voxels, not point
                    # obstacles.  Account for half a voxel when comparing the
                    # obstacle-center distance with the requested footprint;
                    # otherwise a 0.45 m footprint on a 0.5 m grid would have
                    # no lateral inflation at all.
                    if (
                        math.hypot(ox * resolution, oy * resolution)
                        > inflation_m + 0.5 * resolution
                    ):
                        continue
                    if (
                        self.evidence.get((ix + ox, iy + oy, iz), 0)
                        >= self.occupied_threshold
                    ):
                        self._column_cache[cache_key] = OCCUPIED
                        return OCCUPIED
        state = FREE if observed_free else UNKNOWN
        self._column_cache[cache_key] = state
        return state

    def direction_evidence(self, x, y, ground_z, dx, dy, terrain_resolution_m):
        """Return true-frontier evidence along a cardinal or diagonal direction.

        Any inflated obstacle in the look-ahead corridor closes this boundary.
        A fully observed free corridor without terrain support is an observed
        drop/unsupported region, not an exploration frontier.
        """

        terrain_resolution_m = max(self.resolution_m, float(terrain_resolution_m))
        direction_norm = math.hypot(dx, dy)
        if direction_norm <= 1.0e-9:
            return DirectionEvidence(False, 0, 0)
        unit_x, unit_y = float(dx) / direction_norm, float(dy) / direction_norm
        steps = max(1, int(math.ceil(self.frontier_lookahead_m / terrain_resolution_m)))
        center_states = []
        occupied_columns = 0
        for step in range(1, steps + 1):
            state = self.column_state(
                x + unit_x * step * terrain_resolution_m,
                y + unit_y * step * terrain_resolution_m,
                ground_z,
            )
            center_states.append(state)
            if state == OCCUPIED:
                occupied_columns += 1

        # An observed wall inside the short look-ahead is a closed physical
        # boundary. Rejecting the complete corridor also suppresses sparse-ray
        # holes immediately in front of that wall.
        if occupied_columns:
            return DirectionEvidence(False, 0, occupied_columns)
        if UNKNOWN not in center_states:
            return DirectionEvidence(False, 0, 0)

        lateral_steps = int(math.ceil(self.frontier_lateral_m / terrain_resolution_m))
        unknown_columns = 0
        occupied_fan = 0
        perpendicular_x, perpendicular_y = -unit_y, unit_x
        for step in range(1, steps + 1):
            center_x = x + unit_x * step * terrain_resolution_m
            center_y = y + unit_y * step * terrain_resolution_m
            for lateral in range(-lateral_steps, lateral_steps + 1):
                state = self.column_state(
                    center_x + perpendicular_x * lateral * terrain_resolution_m,
                    center_y + perpendicular_y * lateral * terrain_resolution_m,
                    ground_z,
                )
                if state == UNKNOWN:
                    unknown_columns += 1
                elif state == OCCUPIED:
                    occupied_fan += 1
        return DirectionEvidence(
            unknown_columns >= self.frontier_min_unknown_columns,
            unknown_columns,
            occupied_fan,
        )

    def has_clear_line_of_sight(self, start, end, inflation_m=None):
        """Check the inflated robot-clearance columns between two viewpoints."""

        start = np.asarray(start, dtype=np.float64).reshape(3)
        end = np.asarray(end, dtype=np.float64).reshape(3)
        distance = float(np.linalg.norm(end[:2] - start[:2]))
        if distance <= self.resolution_m:
            return True
        steps = max(1, int(math.ceil(distance / self.resolution_m)))
        for step in range(1, steps):
            ratio = float(step) / steps
            point = start + ratio * (end - start)
            if (
                self.column_state(
                    point[0], point[1], point[2], inflation_m
                )
                == OCCUPIED
            ):
                return False
        return True

    def _completed_key(self, x, y, z, dx, dy):
        resolution = self.completed_resolution_m
        return (
            self._round_cell(x, resolution),
            self._round_cell(y, resolution),
            self._round_cell(z, resolution),
            int(dx),
            int(dy),
        )

    def is_completed(self, x, y, z, dx, dy):
        return self._completed_key(x, y, z, dx, dy) in self.completed_regions

    def mark_completed(self, x, y, z, directions, radius_m):
        """Mask a fixed physical disk on one elevation layer and direction."""

        directions = tuple(directions)
        if not directions:
            return 0
        resolution = self.completed_resolution_m
        center = self._completed_key(x, y, z, directions[0][0], directions[0][1])
        radius_m = max(resolution, float(radius_m))
        radius_cells = int(math.ceil(radius_m / resolution))
        vertical_cells = int(
            math.ceil(self.completed_vertical_tolerance_m / resolution)
        )
        before = len(self.completed_regions)
        for ox in range(-radius_cells, radius_cells + 1):
            for oy in range(-radius_cells, radius_cells + 1):
                if math.hypot(ox * resolution, oy * resolution) > radius_m:
                    continue
                for oz in range(-vertical_cells, vertical_cells + 1):
                    for dx, dy in directions:
                        self.completed_regions.add(
                            (center[0] + ox, center[1] + oy, center[2] + oz, dx, dy)
                        )
        added = len(self.completed_regions) - before
        if added:
            self.completed_revision += 1
        return added

    @staticmethod
    def closure_state(
        unresolved_regions,
        search_truncated,
        reachable_cells,
        minimum_reachable_cells,
        map_quiet_s,
        required_map_quiet_s,
    ):
        """Evaluate closure of the currently reachable exploration component."""

        if search_truncated:
            return "SEARCH_TRUNCATED"
        if reachable_cells < minimum_reachable_cells:
            return "INSUFFICIENT_REACHABILITY"
        if unresolved_regions:
            return "OPEN"
        if map_quiet_s < required_map_quiet_s:
            return "SETTLING"
        return "CLOSED"
