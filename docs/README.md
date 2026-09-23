# FLEX-E for AEDE Garage

`flex_e_planner` is a standalone ROS package deployed into the same catkin
workspace as `autonomous_exploration_development_environment` and
`dsv_planner`.  Its external interface intentionally follows the DSV Garage
launch convention:

| DSV package | FLEX-E package |
|---|---|
| `roslaunch dsvp_launch explore_garage.launch` | `roslaunch flex_e_planner explore_garage.launch` |
| `dsvp_launch/config/exploration_garage.yaml` | `flex_e_planner/config/exploration_garage.yaml` |
| `/state_estimation`, `/terrain_map_ext` | same inputs |
| `/way_point`, `/runtime` | same outputs |

FLEX-E does not load the Garage mesh or any offline point cloud.  It constructs
a multi-level observed-support graph from AEDE's `/terrain_map_ext`, admits
only slope/step-valid neighbor edges, selects reachable frontier cells, and
publishes a short observed route prefix to AEDE's existing `/way_point`
interface.  `localPlanner` and `pathFollower` remain the only AEDE components
that generate `/cmd_vel`.

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
`max_slope_deg`, `max_step_m`, `grid_resolution_m`, `z_resolution_m`,
`minimum_goal_distance_m`, and `execution_horizon_m`.  A selected frontier is
kept as a persistent global goal while `execution_horizon_m` controls only the
rolling controller subgoal.  Odometry-rate waypoint crossing detection avoids
commanding a waypoint behind a fast-moving vehicle.  An unreachable or stalled
global goal is temporarily rejected using `rejected_goal_radius_m` and
`rejected_goal_memory_s`.

Keep the AEDE topic names unchanged unless the entire simulator and both
comparison planners are remapped consistently.

## Read-only wiring checks

After a launch, these commands do not publish commands or alter state:

```bash
rostopic info /state_estimation
rostopic info /terrain_map_ext
rostopic info /way_point
rostopic info /cmd_vel
rostopic hz /runtime
```

Expected ownership: AEDE publishes the first two topics, FLEX-E publishes
`/way_point` and `/runtime`, and AEDE's `pathFollower` publishes `/cmd_vel`.
RViz may also advertise `/way_point` for its manual Waypoint tool; that is
normal.  The DSV `/exploration` node must not be listed as a publisher during a
FLEX-E run.

The decision trace should contain `new_global_frontier_gain_*` followed by one
or more `continued_global_frontier` rows.  A long run containing only one fixed
waypoint usually means a second planner is publishing `/way_point`, or the
selected goal is physically blocked.  In the latter case FLEX-E emits
`subgoal stalled`, rejects that frontier, and selects another one after
`active_goal_stall_s`.

The optimized Garage defaults also publish latched `true` messages on
`/exploration_finish` and `/stop_exploring` only after both the local and the
expanded retained-graph frontier searches find no safe novel target for
`no_frontier_finish_s`, or when `max_exploration_s` is reached. The final
waypoint is set to the current vehicle pose so the AEDE controller stops.
The Garage default keeps `shutdown_on_finish: false` so Gazebo and RViz remain
available for coverage inspection. Set it to `true` when batch runs should
close the combined launch and optional rosbag recorder automatically.
Detailed evidence and the rationale for the repeat/edge handling changes are
in [`optimization_analysis.md`](optimization_analysis.md).

`frontier_search_range_m` is the normal 25 m local search radius.
`global_frontier_search_range_m` and `global_max_search_nodes` are used only
when that local search is exhausted; the Garage defaults are 120 m and 30,000
nodes.
