# FLEX-E for AEDE Garage

`flex_e_planner` is a standalone ROS package deployed into the same catkin
workspace as `autonomous_exploration_development_environment` and
`dsv_planner`.  Its external interface intentionally follows the DSV Garage
launch convention:

| DSV package | FLEX-E package |
|---|---|
| `roslaunch dsvp_launch explore_garage.launch` | `roslaunch flex_e_planner explore_garage.launch` |
| `dsvp_launch/config/exploration_garage.yaml` | `flex_e_planner/config/exploration_garage.yaml` |
| `/state_estimation`, `/state_estimation_at_scan`, `/terrain_map_ext`, `/registered_scan` | same inputs |
| `/way_point`, `/runtime` | same outputs |

FLEX-E does not load the Garage mesh or any offline point cloud. It combines a
multi-level observed-support graph from `/terrain_map_ext` with a sparse 3-D
free/occupied/unknown evidence map raycast from `/registered_scan`. A missing
terrain neighbor is accepted as a true frontier only when its clearance
corridor contains useful unknown space and is not closed by an observed wall
or drop. The information target may lie in unknown space, but every executable
`/way_point` remains a safe observed support cell. `localPlanner` and
`pathFollower` remain the only AEDE components that generate `/cmd_vel`.

## Build

```bash
source /opt/ros/noetic/setup.bash
cd /home/dawn/workspace/exploration
catkin_make -DCMAKE_BUILD_TYPE=Release -j2
source devel/setup.bash --extend
```

## Separate AEDE + planner launch (same form as DSV)

Terminal A starts only the standard AEDE Garage system:

```bash
source /opt/ros/noetic/setup.bash
source /home/dawn/workspace/exploration/devel/setup.bash --extend
roslaunch vehicle_simulator system_garage.launch gazebo_gui:=false
```

Terminal B starts only FLEX-E:

```bash
source /opt/ros/noetic/setup.bash
source /home/dawn/workspace/exploration/devel/setup.bash --extend
roslaunch flex_e_planner explore_garage.launch \
  seed:=0 \
  decision_trace_path:=/tmp/flex_e_garage_decisions.csv
```

The DSV-compatible rosbag arguments are also supported.  Set
`enable_bag_record:=true` to save common AEDE inputs, commands and runtime
topics below `~/Desktop/simulation_bags/flex_e/<bag_name>/`.

For a DSV baseline, stop FLEX-E and replace the terminal-B command with:

```bash
roslaunch dsvp_launch explore_garage.launch simulation:=true
```

Never launch FLEX-E and DSV at the same time: both write `/way_point` and
`/runtime`.

## Convenience launch

The following starts the AEDE Garage system and FLEX-E together.  Use it only
for an individual smoke run; use the separated form above for comparisons.

```bash
roslaunch flex_e_planner flex_e_garage.launch \
  gazebo_gui:=false seed:=0 \
  decision_trace_path:=/tmp/flex_e_garage_decisions.csv
```

For unattended automated checks, use the lighter AEDE-only launch (no joystick,
RViz or visualization-only nodes):

```bash
roslaunch flex_e_planner flex_e_garage_headless.launch \
  seed:=0 decision_trace_path:=/tmp/flex_e_garage_decisions.csv
```

## Configuration

Pass an edited copy of `config/exploration_garage.yaml` without changing the
launch file:

```bash
roslaunch flex_e_planner explore_garage.launch \
  planner_param_file:=/absolute/path/to/my_flex_e_garage.yaml \
  seed:=7
```

The most important physical/planning parameters are `vehicle_height_m`,
`max_slope_deg`, `max_step_m`, `grid_resolution_m`, `occupancy_resolution_m`,
`frontier_lookahead_m`, `minimum_goal_distance_m`, and `execution_horizon_m`.
A selected safe viewpoint is kept as a global goal while
`execution_horizon_m` controls its rolling controller subgoal. Reaching it
starts an observation interval. The old point-based rejected-goal list has
been replaced by a fixed position/elevation/direction completion mask: a
no-information or failed region is closed once, while a genuinely advancing
boundary remains eligible farther outward.

Keep the AEDE topic names unchanged unless the entire simulator and both
comparison planners are remapped consistently.

## Read-only wiring checks

After a launch, these commands do not publish commands or alter state:

```bash
rostopic info /state_estimation
rostopic info /state_estimation_at_scan
rostopic info /terrain_map_ext
rostopic info /registered_scan
rostopic info /way_point
rostopic info /cmd_vel
rostopic hz /runtime
rostopic echo -n 1 /flex_e/exploration_closed
```

Expected ownership: AEDE publishes the four input topics, FLEX-E publishes
`/way_point`, `/runtime`, `/flex_e/true_frontiers`,
`/flex_e/safe_viewpoints`, and `/flex_e/exploration_closed`; AEDE's
`pathFollower` publishes `/cmd_vel`.
RViz may also advertise `/way_point` for its manual Waypoint tool; that is
normal.  The DSV `/exploration` node must not be listed as a publisher during a
FLEX-E run.

The decision trace should contain `new_*_true_frontier_*` followed by one or
more `continued_global_frontier` rows. It also records occupancy voxels,
physical completion-mask cells, information gain, reachable cells, transient
frontier regions, closure state and whether the global search was truncated.
Use `/flex_e/true_frontiers` and `/flex_e/safe_viewpoints` as RViz PointCloud2
displays when tuning the evaluator.

The optimized Garage defaults publish latched `true` messages on
`/exploration_finish` and `/stop_exploring` only after an untruncated global
closure audit finds no unresolved true-frontier region, the reachable graph is
large enough, the observed map is quiet, and three consecutive audits agree.
`/flex_e/exploration_closed` is true only for this closure condition, not for a
time-limit stop. The final waypoint is set to the current vehicle pose.
The Garage default keeps `shutdown_on_finish: false` so Gazebo and RViz remain
available for coverage inspection. Set it to `true` when batch runs should
close the combined launch and optional rosbag recorder automatically.
Detailed evidence and the rationale for the repeat/edge handling changes are
in [`optimization_analysis.md`](optimization_analysis.md).

`frontier_search_range_m` is the normal 25 m local search radius.
`global_frontier_search_range_m` and `global_max_search_nodes` are used only
when that local search is exhausted; the Garage defaults are 120 m and 30,000
nodes.
The final closure audit expands this to `completion_search_range_m=200` and
`completion_max_search_nodes=160000`; hitting that limit prevents completion.
