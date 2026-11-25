#!/usr/bin/env python3
"""ROS topic and message adapter for the Python-native FLEX-E planner."""

from __future__ import print_function

import math

import message_filters
import numpy as np
import rospy
from flex_e_core.planner import FlexECore
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool, Float32, Header, String, Time


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


class _RosLogger(object):
    def info(self, message, *args):
        rospy.loginfo(message, *args)

    def warning(self, message, *args):
        rospy.logwarn(message, *args)


class FlexEPlanner(object):
    """Keep AEDE topics and lifecycle while delegating decisions to FlexECore."""

    def __init__(self):
        param = lambda name, default: rospy.get_param("~" + name, default)
        # Planar experiments constrain support points. Full Garage runs leave
        # both limits unset and retain the observed ramps and upper decks.
        self.terrain_z_min = param("terrain_z_min", None)
        self.terrain_z_max = param("terrain_z_max", None)
        self.core = FlexECore(
            param_source=param,
            logger=_RosLogger(),
            debug_callback=self._publish_debug_cloud,
        )
        self.frame_id = self.core.frame_id
        self.last_published_revision = 0
        self.finished_published = False
        self.shutdown_timer = None

        self.waypoint_pub = rospy.Publisher(
            self.core.waypoint_topic, PointStamped, queue_size=5
        )
        self.motion_goal_pub = rospy.Publisher(
            "/flex_e/motion_goal", Time, queue_size=1, latch=True
        )
        self.motion_goal_stamp = rospy.Time()
        self.motion_goal_pub.publish(Time(data=self.motion_goal_stamp))
        self.runtime_pub = rospy.Publisher(
            self.core.runtime_topic, Float32, queue_size=20
        )
        self.finish_pub = rospy.Publisher(
            self.core.finish_topic, Bool, queue_size=1, latch=True
        )
        self.stop_pub = rospy.Publisher(
            self.core.stop_topic, Bool, queue_size=1, latch=True
        )
        self.closed_pub = rospy.Publisher(
            "/flex_e/exploration_closed", Bool, queue_size=1, latch=True
        )
        self.finish_reason_pub = rospy.Publisher(
            "/flex_e/exploration_finish_reason", String, queue_size=1, latch=True
        )
        self.frontier_debug_pub = rospy.Publisher(
            "/flex_e/true_frontiers", PointCloud2, queue_size=1
        )
        self.viewpoint_debug_pub = rospy.Publisher(
            "/flex_e/safe_viewpoints", PointCloud2, queue_size=1
        )
        self.region_debug_pub = rospy.Publisher(
            "/flex_e/unexplored_regions", PointCloud2, queue_size=1
        )
        self.odom_sub = rospy.Subscriber(
            self.core.odom_topic, Odometry, self._odom_callback, queue_size=50
        )
        self.terrain_sub = rospy.Subscriber(
            self.core.terrain_topic, PointCloud2, self._terrain_callback, queue_size=2
        )
        self.scan_sub = message_filters.Subscriber(
            self.core.registered_scan_topic, PointCloud2
        )
        self.scan_odom_sub = message_filters.Subscriber(
            self.core.scan_odom_topic, Odometry
        )
        self.scan_sync = message_filters.ApproximateTimeSynchronizer(
            [self.scan_sub, self.scan_odom_sub], queue_size=20, slop=0.05
        )
        self.scan_sync.registerCallback(self._registered_scan_callback)
        self.timer = rospy.Timer(rospy.Duration(self.core.period_s), self._plan)
        rospy.on_shutdown(self.core._close_trace)
        rospy.loginfo(
            "FLEX-E ready: terrain=%s scan=%s scan_pose=%s odometry=%s waypoint=%s grid=%.2f/%.2f m",
            self.core.terrain_topic,
            self.core.registered_scan_topic,
            self.core.scan_odom_topic,
            self.core.odom_topic,
            self.core.waypoint_topic,
            self.core.grid_m,
            self.core.z_grid_m,
        )

    def _odom_callback(self, message):
        point = message.pose.pose.position
        orientation = message.pose.pose.orientation
        sin_yaw = 2.0 * (
            orientation.w * orientation.z + orientation.x * orientation.y
        )
        cos_yaw = 1.0 - 2.0 * (
            orientation.y * orientation.y + orientation.z * orientation.z
        )
        yaw = math.atan2(sin_yaw, cos_yaw)
        self.core.set_pose(
            (point.x, point.y, point.z),
            yaw,
            message.header.stamp.to_sec(),
        )

    def _terrain_callback(self, message):
        try:
            xyz, intensity = _pointcloud_xyzi(message)
        except ValueError as error:
            rospy.logwarn_throttle(5.0, "FLEX-E terrain decode: %s", error)
            return
        stamp = message.header.stamp.to_sec() or rospy.Time.now().to_sec()
        if self.terrain_z_min is not None or self.terrain_z_max is not None:
            keep = np.ones(len(xyz), dtype=bool)
            if self.terrain_z_min is not None:
                keep &= xyz[:, 2] >= float(self.terrain_z_min)
            if self.terrain_z_max is not None:
                keep &= xyz[:, 2] <= float(self.terrain_z_max)
            xyz, intensity = xyz[keep], intensity[keep]
        self.core.integrate_terrain(xyz, intensity, stamp)

    def _registered_scan_callback(self, scan, scan_odom):
        try:
            xyz, _intensity = _pointcloud_xyzi(scan)
        except ValueError as error:
            rospy.logwarn_throttle(5.0, "FLEX-E registered scan decode: %s", error)
            return
        point = scan_odom.pose.pose.position
        stamp = scan.header.stamp.to_sec() or rospy.Time.now().to_sec()
        self.core.integrate_scan(
            xyz,
            (point.x, point.y, point.z),
            stamp,
        )

    def _publish_debug_cloud(self, kind, points):
        header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id)
        if kind == "regions":
            fields = [
                PointField("x", 0, PointField.FLOAT32, 1),
                PointField("y", 4, PointField.FLOAT32, 1),
                PointField("z", 8, PointField.FLOAT32, 1),
                PointField("intensity", 12, PointField.FLOAT32, 1),
            ]
            self.region_debug_pub.publish(
                point_cloud2.create_cloud(header, fields, points)
            )
        elif kind == "frontiers":
            self.frontier_debug_pub.publish(
                point_cloud2.create_cloud_xyz32(header, points)
            )
        elif kind == "viewpoints":
            self.viewpoint_debug_pub.publish(
                point_cloud2.create_cloud_xyz32(header, points)
            )

    def _publish_waypoint(self, xyz, stamp):
        waypoint = PointStamped()
        waypoint.header.frame_id = self.frame_id
        waypoint.header.stamp = stamp
        waypoint.point.x, waypoint.point.y, waypoint.point.z = xyz
        self.waypoint_pub.publish(waypoint)

    def _hold_motion(self):
        if self.motion_goal_stamp != rospy.Time():
            self.motion_goal_stamp = rospy.Time()
            self.motion_goal_pub.publish(Time(data=self.motion_goal_stamp))
            # Also stop older AEDE launch files without motion-goal support.
            if self.core.vehicle is not None:
                self._publish_waypoint(self.core.vehicle[:3], rospy.Time.now())

    def _on_finished(self, reason):
        """Notify experiment adapters of the core's terminal decision."""

    def _plan(self, _event):
        decision = self.core.step(rospy.Time.now().to_sec())
        runtime_ms = decision["diagnostics"].get("decision_ms")
        if runtime_ms is not None:
            self.runtime_pub.publish(Float32(data=float(runtime_ms) / 1000.0))

        revision = self.core.subgoal_revision
        waypoint = decision["subgoal"]
        if waypoint is None or decision["status"] != "moving":
            self._hold_motion()
        elif revision != self.last_published_revision:
            stamp = rospy.Time.now()
            if stamp != rospy.Time():
                self._publish_waypoint(waypoint, stamp)
                self.motion_goal_stamp = stamp
                self.motion_goal_pub.publish(Time(data=stamp))
                self.last_published_revision = revision

        if decision["status"] == "finished":
            if self.finished_published:
                return
            self.finished_published = True
            reason = decision["diagnostics"]["finish_reason"]
            self._on_finished(reason)
            self.finish_reason_pub.publish(String(data=reason))
            self.finish_pub.publish(Bool(data=True))
            self.stop_pub.publish(Bool(data=True))
            self.closed_pub.publish(
                Bool(data=bool(decision["diagnostics"]["closure_closed"]))
            )
            self.timer.shutdown()
            if self.core.shutdown_on_finish:
                self.shutdown_timer = rospy.Timer(
                    rospy.Duration(self.core.finish_shutdown_delay_s),
                    lambda _event: rospy.signal_shutdown(
                        "exploration finished: " + reason
                    ),
                    oneshot=True,
                )
            return

        self.closed_pub.publish(Bool(data=False))


def main():
    rospy.init_node("flex_e_planner")
    FlexEPlanner()
    rospy.spin()


if __name__ == "__main__":
    main()
