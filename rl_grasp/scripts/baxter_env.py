#!/usr/bin/env python

import argparse
import struct
import sys
import copy

import rospy
import rospkg

from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import (
    Header,
    Empty,
)

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

import baxter_interface
import rospy

# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

# Instantiate CvBridge
bridge = CvBridge()

FAIL_REWARD = -1
SUCCESS_REWARD = 1

class BaxterEnv(object):
    def __init__(self, limb, hover_distance = 0.15, verbose=True):
        self._limb_name = limb # string
        self._hover_distance = hover_distance # in meters
        self._verbose = verbose # bool
        self._limb = baxter_interface.Limb(limb)
        self.joint_space = len(self._limb.joint_angles())
        self._gripper = baxter_interface.Gripper(limb)
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)
        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        self.ctr=0
        self.canSubscriber = True

    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(start_angles)
        self.gripper_open()
        rospy.sleep(1.0)
        print("Running. Ctrl-c to quit")

    def ik_request(self, pose):
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        try:
            resp = self._iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False
        # Check if result valid, and type of seed ultimately used to get solution
        # convert rospy's string representation of uint8[]'s to int's
        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        limb_joints = {}
        if (resp_seeds[0] != resp.RESULT_INVALID):
            seed_str = {
                        ikreq.SEED_USER: 'User Provided Seed',
                        ikreq.SEED_CURRENT: 'Current Joint Angles',
                        ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                       }.get(resp_seeds[0], 'None')
            if self._verbose:
                print("IK Solution SUCCESS - Valid Joint Solution Found from Seed Type: {0}".format(
                         (seed_str)))
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            if self._verbose:
                print("IK Joint Solution:\n{0}".format(limb_joints))
                print("------------------")
        else:
            rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
            return False
        return limb_joints

    def _guarded_move_to_joint_position(self, joint_angles):
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")

    def gripper_open(self):
        self._gripper.open()
        rospy.sleep(1.0)

    def gripper_close(self):
        self._gripper.close()
        rospy.sleep(1.0)
        #reward_function() # set reward value for current grasp

    def _approach(self, pose):
        approach = copy.deepcopy(pose)
        # approach with a pose the hover-distance above the requested pose
        approach.position.z = approach.position.z + self._hover_distance
        joint_angles = self.ik_request(approach)
        self._guarded_move_to_joint_position(joint_angles)

    def _retract(self):
        # retrieve current pose from endpoint
        current_pose = self._limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x
        ik_pose.position.y = current_pose['position'].y
        ik_pose.position.z = current_pose['position'].z + self._hover_distance
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
        joint_angles = self.ik_request(ik_pose)
        # servo up from current pose
        self._guarded_move_to_joint_position(joint_angles)

    def _servo_to_pose(self, pose):
        # servo down to release
        joint_angles = self.ik_request(pose)
        self._guarded_move_to_joint_position(joint_angles)

    def pick(self, pose):
        # open the gripper
        self.gripper_open()
        # servo above pose
        self._approach(pose)
        # servo to pose
        self._servo_to_pose(pose)
        # close gripper
        self.gripper_close()
        # retract to clear object
        self._retract()

    def place(self, pose):
        # servo above pose
        self._approach(pose)
        # servo to pose
        self._servo_to_pose(pose)
        # open the gripper
        self.gripper_open()
        # retract to clear object
        self._retract()

    def image_callback(self, msg):
        #print("Received an image!")
        if self.canSubscriber is True:
            try:
                # Convert your ROS Image message to OpenCV2
                cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError, e:
                print(e)
            else:
                # Save your OpenCV2 image as a jpeg
                cv2.imwrite('full_image.jpeg', cv2_img)
                cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
                cv2_img = cv2_img[150:300,350:500]
                cv2_img = cv2.resize(cv2_img, (84, 84))
                cv2.imwrite('image.jpeg', cv2_img)

    def listener(self):
        image_topic = "/cameras/left_hand_camera/image"
        rospy.Subscriber(image_topic, Image, self.image_callback)

    def reward_function(self):
        if self._gripper.position() < 10:
            return FAIL_REWARD #failed
        else:
            return SUCCESS_REWARD # #succeeded

    def reset(self, starting_joint_angles):
        self.canSubscriber = True
        self.move_to_start(starting_joint_angles)
        self.canSubscriber = False
        #img = cv2.imread('~/.ros/image_%d.jpeg'%self.ctr,0)
        #img = cv2.imread('/home/gvimlan/.ros/image.jpeg',0)
        img = cv2.imread('/home/gvimlan/.ros/image.jpeg', cv2.COLOR_BGR2GRAY)
        if img is None:
            while(True):
                self.canSubscriber = True
                rospy.sleep(1.0)
                self.canSubscriber = False
                img = cv2.imread('/home/gvimlan/.ros/image.jpeg', cv2.COLOR_BGR2GRAY)
                if img is not None:
                    break
        self.ctr+=1
        return img

    def render():
        pass

    def step(self,block_pose):
        #check if joint angles is valid or else program will crash
        # if bad joint angles then return reward = 0
        joint_angles = self.ik_request(block_pose)
        if joint_angles == False:
            return  FAIL_REWARD, 1, {}
        print("\nPicking...")
        self.pick(block_pose)
        reward = self.reward_function()
        print("\nPlacing...")
        self.place(block_pose)
        print('Pick and Place completed')
        print('REWARD')
        print(reward)
        return reward, 1, {}
