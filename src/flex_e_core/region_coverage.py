"""Persistent physical-subspace coverage state for FLEX-E.

The planner's geometric frontier clusters are intentionally transient: their
membership changes whenever the terrain map grows. This module keeps the
longer-lived exploration state on fixed XYZ subspaces instead. The state
machine follows TARE's global representation while adding retryable
``DEFERRED`` and bounded ``INACCESSIBLE`` states so neither one controller
failure nor one disconnected historical subspace can deadlock exploration.
"""

from __future__ import division

import math


UNSEEN = "UNSEEN"
EXPLORING = "EXPLORING"
COVERED = "COVERED"
DEFERRED = "DEFERRED"
INACCESSIBLE = "INACCESSIBLE"


class RegionState(object):
    __slots__ = (
        "status",
        "frontier_points",
        "covered_cells",
        "total_cells",
        "coverage_ratio",
        "selectable",
        "empty_audits",
        "reopen_audits",
        "failures",
        "defer_until_s",
        "last_update_s",
        "last_audit_s",
        "last_selected_s",
        "covered_total_cells",
        "unselectable_audits",
        "last_unselectable_audit_s",
        "orphan_audits",
        "last_orphan_audit_s",
    )

    def __init__(self):
        self.status = UNSEEN
        self.frontier_points = 0
        self.covered_cells = 0
        self.total_cells = 0
        self.coverage_ratio = 0.0
        self.selectable = 0
        self.empty_audits = 0
        self.reopen_audits = 0
        self.failures = 0
        self.defer_until_s = 0.0
        self.last_update_s = 0.0
        self.last_audit_s = -float("inf")
        self.last_selected_s = -float("inf")
        self.covered_total_cells = 0
        self.unselectable_audits = 0
        self.last_unselectable_audit_s = -float("inf")
        self.orphan_audits = 0
        self.last_orphan_audit_s = -float("inf")


class RegionCoverageTracker(object):
    """Track unfinished exploration evidence on fixed physical subspaces."""

    def __init__(
        self,
        xy_size_m=8.0,
        height_m=3.0,
        open_frontier_points=1,
        close_frontier_points=1,
        reopen_frontier_points=2,
        coverage_complete_ratio=0.92,
        coverage_min_uncovered_cells=6,
        reopen_new_cells=12,
        unselectable_close_audits=2,
        blocked_frontier_audits=3,
        orphan_close_audits=3,
        close_audits=3,
        reopen_audits=2,
        defer_retry_s=30.0,
        max_failures=3,
    ):
        self.xy_size_m = max(1.0, float(xy_size_m))
        self.height_m = max(0.5, float(height_m))
        self.open_frontier_points = max(1, int(open_frontier_points))
        self.close_frontier_points = max(1, int(close_frontier_points))
        self.reopen_frontier_points = max(
            self.open_frontier_points, int(reopen_frontier_points)
        )
        self.coverage_complete_ratio = min(
            1.0, max(0.50, float(coverage_complete_ratio))
        )
        self.coverage_min_uncovered_cells = max(
            1, int(coverage_min_uncovered_cells)
        )
        self.reopen_new_cells = max(1, int(reopen_new_cells))
        self.unselectable_close_audits = max(1, int(unselectable_close_audits))
        self.blocked_frontier_audits = max(1, int(blocked_frontier_audits))
        self.orphan_close_audits = max(1, int(orphan_close_audits))
        self.close_audits = max(1, int(close_audits))
        self.reopen_audits = max(1, int(reopen_audits))
        self.defer_retry_s = max(1.0, float(defer_retry_s))
        self.max_failures = max(1, int(max_failures))
        self.regions = {}

    @staticmethod
    def _cell(value, resolution):
        return int(math.floor(float(value) / resolution))

    def region_key(self, x, y, z):
        return (
            self._cell(x, self.xy_size_m),
            self._cell(y, self.xy_size_m),
            self._cell(z, self.height_m),
        )

    def region_center(self, key):
        return (
            (key[0] + 0.5) * self.xy_size_m,
            (key[1] + 0.5) * self.xy_size_m,
            (key[2] + 0.5) * self.height_m,
        )

    def _get(self, key):
        state = self.regions.get(key)
        if state is None:
            state = RegionState()
            self.regions[key] = state
        return state

    def update(self, evidence, observed_regions, now_s):
        """Update only subspaces covered by the current graph search.

        ``evidence`` maps a region key to counts named ``frontier_points``,
        ``covered_cells``, ``total_cells`` and ``selectable``. Completion is
        therefore based on observed traversable space, not on distance to
        obstacle surfaces. Not touching regions outside ``observed_regions``
        prevents a short local search from closing a distant subspace.
        """

        observed = set(observed_regions)
        observed.update(evidence)
        for key in observed:
            values = evidence.get(key, {})
            frontier = max(0, int(values.get("frontier_points", 0)))
            covered = max(0, int(values.get("covered_cells", 0)))
            total = max(0, int(values.get("total_cells", 0)))
            covered = min(covered, total)
            selectable = max(0, int(values.get("selectable", 0)))
            state = self.regions.get(key)
            if state is None and not (frontier or total or selectable):
                continue
            state = self._get(key)
            state.frontier_points = frontier
            state.covered_cells = covered
            state.total_cells = total
            state.coverage_ratio = float(covered) / total if total else 0.0
            state.selectable = selectable
            if selectable:
                state.unselectable_audits = 0
            state.last_update_s = float(now_s)
            state.orphan_audits = 0
            new_audit = float(now_s) > state.last_audit_s + 1.0e-6
            if new_audit:
                state.last_audit_s = float(now_s)

            uncovered = max(0, total - covered)
            coverage_open = (
                uncovered >= self.coverage_min_uncovered_cells
                and state.coverage_ratio < self.coverage_complete_ratio
            )
            is_open = frontier >= self.open_frontier_points or coverage_open
            coverage_complete = (
                total > 0
                and state.coverage_ratio >= self.coverage_complete_ratio
            )
            is_complete = (
                frontier < self.close_frontier_points
                and coverage_complete
            )
            # A stable feasible frontier must be able to reopen a room before
            # the robot enters it; requiring graph growth as well creates a
            # causal deadlock. Significant graph growth is an independent
            # reason to audit the region again.
            is_strong = (
                (
                    frontier >= self.reopen_frontier_points
                    and selectable > 0
                )
                or total >= state.covered_total_cells + self.reopen_new_cells
            )

            if state.status == DEFERRED:
                if now_s < state.defer_until_s:
                    continue
                state.status = EXPLORING
                if is_open:
                    state.empty_audits = 0

            if state.status == UNSEEN:
                if is_open:
                    state.status = EXPLORING
                    state.empty_audits = 0
                elif total:
                    # A fully visible room still gets the configured number
                    # of stable closure audits, but never produces a waypoint
                    # merely to approach its boundary.
                    state.status = EXPLORING
                    state.empty_audits = 1 if is_complete and new_audit else 0
                    if state.empty_audits >= self.close_audits:
                        state.status = COVERED
                        state.covered_total_cells = total
                continue

            if state.status == EXPLORING:
                state.reopen_audits = 0
                if is_complete:
                    if new_audit:
                        state.empty_audits += 1
                    if state.empty_audits >= self.close_audits:
                        state.status = COVERED
                        state.failures = 0
                        state.covered_total_cells = total
                else:
                    state.empty_audits = 0
                continue

            if state.status in (COVERED, INACCESSIBLE):
                state.empty_audits = 0
                if is_strong:
                    if new_audit:
                        state.reopen_audits += 1
                    if state.reopen_audits >= self.reopen_audits:
                        state.status = EXPLORING
                        state.reopen_audits = 0
                else:
                    state.reopen_audits = 0

    def mark_selected(self, key, now_s):
        state = self._get(key)
        state.status = EXPLORING
        state.last_selected_s = float(now_s)
        state.empty_audits = 0
        state.unselectable_audits = 0

    def mark_progress(self, key):
        if key is None:
            return
        state = self._get(key)
        state.status = EXPLORING
        state.empty_audits = 0
        state.failures = max(0, state.failures - 1)
        state.unselectable_audits = 0

    def mark_covered(self, key, total_cells=None):
        if key is None:
            return
        state = self._get(key)
        state.status = COVERED
        state.empty_audits = self.close_audits
        state.reopen_audits = 0
        state.failures = 0
        state.unselectable_audits = 0
        if total_cells is not None:
            state.total_cells = max(0, int(total_cells))
        state.covered_total_cells = state.total_cells

    def defer(self, key, now_s):
        if key is None:
            return 0
        state = self._get(key)
        state.status = DEFERRED
        state.failures += 1
        multiplier = min(4, state.failures)
        state.defer_until_s = float(now_s) + self.defer_retry_s * multiplier
        state.empty_audits = 0
        return state.failures

    def resolve_unselectable(self, regions, now_s):
        """Bound OPEN-without-a-target instead of leaving it alive forever.

        A frontier-free region with no informative interior viewpoint contains
        only occluded/sampling residuals and is closed after a short audit. A
        region that repeatedly has raw frontier evidence but no safe viewpoint
        is classified INACCESSIBLE under the current robot footprint. Both may
        reopen later through stable frontier evidence or graph growth.
        """

        changed = []
        for key in set(regions):
            state = self.regions.get(key)
            if state is None or state.status != EXPLORING:
                continue
            if state.selectable:
                state.unselectable_audits = 0
                continue
            if float(now_s) <= state.last_unselectable_audit_s + 1.0e-6:
                continue
            state.last_unselectable_audit_s = float(now_s)
            state.unselectable_audits += 1
            if state.frontier_points:
                if state.unselectable_audits >= self.blocked_frontier_audits:
                    state.status = INACCESSIBLE
                    state.covered_total_cells = state.total_cells
                    changed.append((key, INACCESSIBLE))
            elif state.unselectable_audits >= self.unselectable_close_audits:
                self.mark_covered(key, state.total_cells)
                changed.append((key, COVERED))
        return changed

    def resolve_orphans(self, reachable_regions, now_s):
        """Bound stale unresolved states outside a full reachable audit."""

        reachable = set(reachable_regions)
        changed = []
        for key, state in self.regions.items():
            if key in reachable or state.status not in (EXPLORING, DEFERRED):
                continue
            if float(now_s) <= state.last_orphan_audit_s + 1.0e-6:
                continue
            state.last_orphan_audit_s = float(now_s)
            state.orphan_audits += 1
            if state.orphan_audits >= self.orphan_close_audits:
                state.status = INACCESSIBLE
                state.covered_total_cells = state.total_cells
                changed.append((key, INACCESSIBLE))
        return changed

    def status(self, key, now_s=None):
        state = self.regions.get(key)
        if state is None:
            return UNSEEN
        if state.status == DEFERRED and now_s is not None and now_s >= state.defer_until_s:
            return EXPLORING
        return state.status

    def unresolved_count(self, regions=None, now_s=None):
        allowed = None if regions is None else set(regions)
        return sum(
            (allowed is None or key in allowed)
            and self.status(key, now_s) in (EXPLORING, DEFERRED)
            for key, state in self.regions.items()
        )

    def status_counts(self, regions=None):
        allowed = None if regions is None else set(regions)
        counts = {
            UNSEEN: 0,
            EXPLORING: 0,
            COVERED: 0,
            DEFERRED: 0,
            INACCESSIBLE: 0,
        }
        for key, state in self.regions.items():
            if allowed is not None and key not in allowed:
                continue
            counts[state.status] += 1
        return counts

    def route(self, choices, start_position, preferred_region=None):
        """Return a coarse open tour through every selectable unfinished cell.

        ``choices`` maps region keys to ``(position_xyz, path_cost)``. A
        nearest-neighbour tour is followed by deterministic 2-opt refinement.
        The first region uses graph path cost; later legs use subspace-center
        distance, avoiding an expensive all-pairs graph search.
        """

        eligible = {
            key: value
            for key, value in choices.items()
            if self.status(key) == EXPLORING
        }
        if not eligible:
            return []
        if preferred_region in eligible:
            first = preferred_region
        else:
            first = min(
                eligible,
                key=lambda key: (
                    eligible[key][1],
                    sum(
                        (eligible[key][0][axis] - start_position[axis]) ** 2
                        for axis in range(3)
                    ),
                    key,
                ),
            )
        route = [first]
        remaining = set(eligible) - {first}
        while remaining:
            current = eligible[route[-1]][0]
            next_key = min(
                remaining,
                key=lambda key: (
                    math.sqrt(
                        sum(
                            (eligible[key][0][axis] - current[axis]) ** 2
                            for axis in range(3)
                        )
                    ),
                    key,
                ),
            )
            route.append(next_key)
            remaining.remove(next_key)

        def distance(first_key, second_key):
            first_pos, second_pos = eligible[first_key][0], eligible[second_key][0]
            return math.sqrt(sum((first_pos[i] - second_pos[i]) ** 2 for i in range(3)))

        improved = True
        while improved and len(route) > 3:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    old = distance(route[i - 1], route[i]) + distance(route[j], route[j + 1])
                    new = distance(route[i - 1], route[j]) + distance(route[i], route[j + 1])
                    if new + 1.0e-6 < old:
                        route[i : j + 1] = reversed(route[i : j + 1])
                        improved = True
        return route
