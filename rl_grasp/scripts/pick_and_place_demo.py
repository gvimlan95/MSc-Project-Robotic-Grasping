#!/usr/bin/env python

import argparse
import struct
import sys
import copy

import rospy
import rospkg

from baxter_env import BaxterEnv
import cv2
import tflearn
import numpy as np
from ddpg import RLBrain
import time


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


def load_gazebo_models(table_pose=Pose(position=Point(x=1.0, y=0.0, z=0.0)),
                       table_reference_frame="world",
                       block_pose=Pose(position=Point(x=0.6725, y=0.1265, z=0.7825)), # x = 1.0
                       block_reference_frame="world"):
    # Get Models' Path
    model_path = rospkg.RosPack().get_path('baxter_sim_examples')+"/models/"
    # Load Table SDF
    table_xml = ''
    with open (model_path + "cafe_table/model.sdf", "r") as table_file:
        table_xml=table_file.read().replace('\n', '')
    # Load Block URDF
    block_xml = ''
    with open (model_path + "block/model.urdf", "r") as block_file:
        block_xml=block_file.read().replace('\n', '')
    # Spawn Table SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_sdf = spawn_sdf("cafe_table", table_xml, "/",
                             table_pose, table_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Block URDF
    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    try:
        spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        resp_urdf = spawn_urdf("block", block_xml, "/",
                               block_pose, block_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn URDF service call failed: {0}".format(e))

def delete_gazebo_models():
    # This will be called on ROS Exit, deleting Gazebo models
    # Do not wait for the Gazebo Delete Model service, since
    # Gazebo should already be running. If the service is not
    # available since Gazebo has been killed, it is fine to error out
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("cafe_table")
        resp_delete = delete_model("block")
    except rospy.ServiceException, e:
        rospy.loginfo("Delete Model service call failed: {0}".format(e))

def main():
    rospy.init_node("pick_and_place")
    # Load Gazebo Models via Spawning Services
    # Note that the models reference is the /world frame
    # and the IK operates with respect to the /base frame
    load_gazebo_models()
    # Remove models from the scene on shutdown
    rospy.on_shutdown(delete_gazebo_models)

    # Wait for the All Clear from emulator startup
    rospy.wait_for_message("/robot/sim/started", Empty)

    limb = 'left'
    hover_distance = 0.15 # meters
    baxterSimulation = BaxterEnv(limb, hover_distance)

    # Starting Joint angles for left arm
    starting_joint_angles = {'left_w0': 0.6699952259595108,
                             'left_w1': 1.030009435085784,
                             'left_w2': -0.4999997247485215,
                             'left_e0': -1.189968899785275,
                             'left_e1': 1.9400238130755056,
                             'left_s0': -0.08000397926829805,
                             'left_s1': -0.9999781166910306}


    # An orientation for gripper fingers to be overhead and parallel to the obj
    overhead_orientation =Quaternion(
                             x=-0.1249590815779,
                             y=0.999649402929,
                             z=0.01737916180073,
                             w=0.00486450832011)
    # block_poses = list()
    # The Pose of the block in its initial location.
    # You may wish to replace these poses with estimates
    # from a perception node.
    block_pose = Pose(position=Point(x=0.7, y=0.15, z=-0.129),orientation=overhead_orientation)
    # Feel free to add additional desired poses for the object.
    # Each additional pose will get its own pick and place.
    # block_poses.append(Pose(
    #     position=Point(x=0.75, y=0.0, z=-0.129),
    #     orientation=overhead_orientation))
    # Move to the desired starting angles
    baxterSimulation.listener()
    rl_brain = RLBrain(baxterSimulation, starting_joint_angles, block_pose)

    #observation = baxterSimulation.reset(starting_joint_angles)
    while not rospy.is_shutdown():
        rl_brain.run(delete_gazebo_models,load_gazebo_models)
        #block_pose.orientation = Quaternion(x=o_orientation[0], y=o_orientation[1], z=o_orientation[2], w=o_orientation[3])
        #_,reward,terminal,_ = baxterSimulation.step(block_pose)
        #nextObservation = baxterSimulation.reset(starting_joint_angles)
        #o_orientation = rl_brain.generate_gripper_values(image)
        #block_poses[idx].orientation = Quaternion(x=o_orientation[0], y=o_orientation[1], z=o_orientation[2], w=o_orientation[3])
        #_, r, r1, _ = baxterSimulation.step(block_pose)
        #print("\nPicking...")
        #baxterSimulation.pick(block_poses[idx])
        #print("\nPlacing...")
        #idx = (idx+1) % len(block_poses)
        #baxterSimulation.place(block_poses[idx])
    return 0

if __name__ == '__main__':
    sys.exit(main())
