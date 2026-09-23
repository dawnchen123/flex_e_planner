# FLEX-E Garage optimization analysis

## Source run

The supplied trace `/tmp/flex_e_garage_fixed.csv` contains 20,150 decision
rows covering 5,144.6 seconds of simulation time.  The corresponding ROS log
is `~/.ros/log/2d9cbab6-b6f7-11f1-bbf5-63321b948a17/`.

Observed failure indicators:

| Metric | Original persistent-goal version |
|---|---:|
| Global frontiers selected | 288 |
| Reached global frontiers | 187 |
| Stalled global frontiers | 99 |
| No longer graph-reachable | 1 |
| Stall fraction | 34.4% |
| Final graph cells | 80,476 |
| Mean new-goal computation | 757.9 ms |
| Maximum new-goal computation | 1,699.8 ms |
| Holding rows | 18,649 (92.6%) |

The main causes were an O(number-of-frontiers x path-history) visited test,
preference for sparsely supported `gain=3/4` boundary cells, navigation all the
way to the raw map frontier, and no completion state. A later 427-second
validation also showed that the initial completion logic searched only the
25 m local radius; it could therefore mistake a locally exhausted corridor for
map completion.

## Implemented changes

1. A level-aware spatial visited index replaces the linear path-history scan.
   The height key keeps overlapping Garage levels independent.
2. Frontier candidates require graph support and prefer continuous `gain=1`
   boundaries. Both the raw frontier and its inward safety viewpoint must be
   novel with respect to the travelled trajectory.
3. The controller stops 2.5 m before the raw unknown boundary. Reached and
   failed frontier anchors are retired spatially for the configured run-long
   rejection interval so the same boundary is not selected repeatedly.
4. A stalled local subgoal is retained for 300 seconds with a 2 m exclusion
   radius, preventing immediate retries through the same obstructed area.
5. Progress uses a 0.25 m deadband, an 8-second no-progress timeout, and an
   18-second hard subgoal timeout.
6. Exploration finishes after 120 seconds without a safe novel frontier (after
   a 120-second minimum run), or at the 3,600-second safety limit. Completion
   publishes `/exploration_finish` and `/stop_exploring` and commands the
   current pose as the final waypoint. The default leaves Gazebo/RViz running
   for inspection; automatic launch shutdown remains configurable.
7. Holding records are sampled every two seconds and are no longer published
   as `/runtime`, so runtime statistics contain actual planning operations.
8. Local-frontier exhaustion now triggers a 120 m / 30,000-node search over
   the retained observed-support graph. Completion begins only if this global
   fallback is also exhausted continuously.

## Short validation

In the first 140-second optimized smoke run, 14 new-goal searches averaged
30.4 ms with a 57.9 ms maximum, versus 757.9/1,699.8 ms in the supplied long
run.  A second validation after viewpoint-novelty filtering produced 9 new-goal
searches averaging 31.8 ms with a 43.2 ms maximum.  These short runs validate
the code path and latency improvement; the next full Garage run is needed to
measure final coverage, completion time, and long-run stall fraction.
