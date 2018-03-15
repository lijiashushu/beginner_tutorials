#!/usr/bin/env python
#coding=utf-8

import rospy
import roslib

import cv;
import cv2;
import cv_bridge

import numpy
import math
import os
import sys
import string
import time
import random
import tf
from sensor_msgs.msg import Image
import baxter_interface
from moveit_commander import conversions
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header
import std_srvs.srv
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest

# directory used to save analysis images
image_directory = os.getenv("HOME") + "/Golf/"

class canny_image_test(object):
    def __init__(self, arm):

        global image_directory
        self.image_dir = image_directory
        # arm ("left" or "right")
        self.limb           = arm
        self.limb_interface = baxter_interface.Limb(self.limb)

        if arm == "left":
            self.other_limb = "right"
        else:
            self.other_limb = "left"
        self.other_limb_interface = baxter_interface.Limb(self.other_limb)



        # start positions  关于球的xyz和关于托盘的xyz

        self.golf_ball_x = 0.65#0.50  # x     = front back
        self.golf_ball_y = 0.4#0.4#0.3  # y     = left right
        self.golf_ball_z = 0.155#0.035  # z     = up down

        '''
        self.golf_ball_x = 0.557  # 0.50  # x     = front back
        self.golf_ball_y = -0.076  # 0.3  # y     = left right
        self.golf_ball_z = -0.005  # 0.035  # z     = up down
        '''

        self.roll = -1.0 * math.pi  # roll  = horizontal  -pi
        self.pitch = 0.0 * math.pi  # pitch = vertical
        self.yaw =0.00# yaw   = rotation

        self.pose = [self.golf_ball_x, self.golf_ball_y, self.golf_ball_z, \
                     self.roll, self.pitch, self.yaw]  # 来回移动，先设置到球的

        # Hough circle accumulator threshold and minimum radius.  #找圆算法
        self.hough_accumulator = 40
        self.hough_min_radius = 15
        self.hough_max_radius = 45

        self.width = 960
        self.height = 600

        # callback image
        self.cv_image = cv.CreateImage((self.width, self.height), 8, 3)

        # canny image  #用来判断边界的图
        self.canny = cv.CreateImage((self.width, self.height), 8, 1)

        # Enable the actuators
        baxter_interface.RobotEnable().enable()

        # set speed as a ratio of maximum speed
        self.limb_interface.set_joint_position_speed(0.5)
        self.other_limb_interface.set_joint_position_speed(0.5)

        print("Camera Setting...\n")
        #reset cameras
        # self.reset_cameras()
        #
        # # close all cameras
        # self.close_camera("left")
        # self.close_camera("right")
        # self.close_camera("head")
        #
        # # open required camera
        # self.open_camera(self.limb, self.width, self.height)

        # subscribe to required camera
        self.subscribe_to_camera(self.limb)

        print("camera OK!!!!!\n")
        # move other arm out of harms way
        if arm == "left":
            self.baxter_ik_move("right", (0.25, -0.50, 0.2, math.pi, 0.0, 0.0))
        else:
            self.baxter_ik_move("left", (0.25, 0.50, 0.2, math.pi, 0.0, 0.0))

    # reset all cameras (incase cameras fail to be recognised on boot)
    def reset_cameras(self):
        reset_srv = rospy.ServiceProxy('cameras/reset', std_srvs.srv.Empty)
        rospy.wait_for_service('cameras/reset', timeout=10)
        reset_srv()

    # open a camera and set camera parameters
    def open_camera(self, camera, x_res, y_res):
        if camera == "left":
            cam = baxter_interface.camera.CameraController("left_hand_camera")
        elif camera == "right":
            cam = baxter_interface.camera.CameraController("right_hand_camera")
        elif camera == "head":
            cam = baxter_interface.camera.CameraController("head_camera")
        else:
            sys.exit("ERROR - open_camera - Invalid camera")

        # close camera
        #cam.close()

        # set camera parameters
        cam.resolution          = (960,600)
        cam.exposure            = -1             # range, 0-100 auto = -1
        cam.gain                = -1             # range, 0-79 auto = -1
        cam.white_balance_blue  = -1             # range 0-4095, auto = -1
        cam.white_balance_green = -1             # range 0-4095, auto = -1
        cam.white_balance_red   = -1             # range 0-4095, auto = -1

        # open camera
        cam.open()

    # close a camera
    def close_camera(self, camera):
        if camera == "left":
            cam = baxter_interface.camera.CameraController("left_hand_camera")
        elif camera == "right":
            cam = baxter_interface.camera.CameraController("right_hand_camera")
        elif camera == "head":
            cam = baxter_interface.camera.CameraController("head_camera")
        else:
            sys.exit("ERROR - close_camera - Invalid camera")

        # set camera parameters to automatic
        cam.exposure            = -1             # range, 0-100 auto = -1
        cam.gain                = -1             # range, 0-79 auto = -1
        cam.white_balance_blue  = -1             # range 0-4095, auto = -1
        cam.white_balance_green = -1             # range 0-4095, auto = -1
        cam.white_balance_red   = -1             # range 0-4095, auto = -1

        # close camera
        cam.close()

    # left camera call back function
    def left_camera_callback(self, data):
        self.camera_callback(data, "Left Hand Camera")

    # right camera call back function
    def right_camera_callback(self, data):
        self.camera_callback(data, "Right Hand Camera")

    # head camera call back function
    def head_camera_callback(self, data):
        self.camera_callback(data, "Head Camera")

    # create subscriber to the required camera
    def subscribe_to_camera(self, camera):
        if camera == "left":
            callback = self.left_camera_callback
            camera_str = "/cameras/left_hand_camera/image"
        elif camera == "right":
            callback = self.right_camera_callback
            camera_str = "/cameras/right_hand_camera/image"
        elif camera == "head":
            callback = self.head_camera_callback
            camera_str = "/cameras/head_camera/image"
        else:
            sys.exit("ERROR - subscribe_to_camera - Invalid camera")

        camera_sub = rospy.Subscriber(camera_str, Image, callback)

    # camera call back function
    def camera_callback(self, data, camera_name):
        # Convert image from a ROS image message to a CV image
        try:
            self.cv_image = cv_bridge.CvBridge().imgmsg_to_cv2(data, "bgr8")

        except cv_bridge.CvBridgeError, e:
            print e

        # 3ms wait
        cv.WaitKey(3)

    def baxter_ik_move(self, limb, rpy_pose):
        #quaternion 四元数 1，i,j,k 代表旋转
        quaternion_pose = conversions.list_to_pose_stamped(rpy_pose, "base")

        node = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        ik_service = rospy.ServiceProxy(node, SolvePositionIK)
        ik_request = SolvePositionIKRequest()
        hdr = Header(stamp=rospy.Time.now(), frame_id="base")

        ik_request.pose_stamp.append(quaternion_pose)
        try:
            rospy.wait_for_service(node, 15.0) #5改成了15
            ik_response = ik_service(ik_request)
        except (rospy.ServiceException, rospy.ROSException), error_message:
            rospy.logerr("Service request failed: %r" % (error_message,))
            sys.exit("ERROR - baxter_ik_move - Failed to append pose")

        if ik_response.isValid[0]:
            print("PASS: Valid joint configuration found")
            # convert response to joint position control dictionary
            limb_joints = dict(zip(ik_response.joints[0].name, ik_response.joints[0].position))
            # move limb
            if self.limb == limb:
                self.limb_interface.move_to_joint_positions(limb_joints)
            else:
                self.other_limb_interface.move_to_joint_positions(limb_joints)
        else:
            # display invalid move message on head display
            #self.splash_screen("Invalid", "move")
            # little point in continuing so exit with error message
            print "requested move =", rpy_pose
            sys.exit("ERROR - baxter_ik_move - No valid joint configuration found")

        if self.limb == limb:               # if working arm
            quaternion_pose = self.limb_interface.endpoint_pose()
            position        = quaternion_pose['position']

            # if working arm remember actual (x,y) position achieved
            self.pose = [position[0], position[1],                                \
                         self.pose[2], self.pose[3], self.pose[4], self.pose[5]]

            # Convert cv image to a numpy array  转化成numpy数组

    def cv2array(self, im):
        depth2dtype = {cv.IPL_DEPTH_8U: 'uint8',
                       cv.IPL_DEPTH_8S: 'int8',
                       cv.IPL_DEPTH_16U: 'uint16',
                       cv.IPL_DEPTH_16S: 'int16',
                       cv.IPL_DEPTH_32S: 'int32',
                       cv.IPL_DEPTH_32F: 'float32',
                       cv.IPL_DEPTH_64F: 'float64'}

        arrdtype = im.depth
        a = numpy.fromstring(im.tostring(),
                             dtype=depth2dtype[im.depth],
                             count=im.width * im.height * im.nChannels)
        a.shape = (im.height, im.width, im.nChannels)

        return a

    def canny_it(self):

        # create gray scale image of balls
        gray_image = cv.CreateImage((self.width, self.height), 8, 1)

        rospy.sleep(1)
        trans = cv.fromarray(self.cv_image)

        # （src，dst，模式：其中当code选用CV_BGR2GRAY时，dst需要是单通道图片）
        cv.CvtColor(trans, gray_image, cv.CV_BGR2GRAY)

        # create gray scale array of balls
        gray_array = self.cv2array(gray_image)

        # create a canny edge detection map of the greyscale image
        cv.Canny(gray_image, self.canny, 40, 200, 3)

        cv.ShowImage('canny',self.canny)

        # save Canny image
        file_name = self.image_dir + "egg_tray_canny.jpg"
        cv.SaveImage(file_name, self.canny)

        # 3ms wait
        cv.WaitKey(0)


def main():
    rospy.init_node("canny_test", anonymous=True)
    test = canny_image_test('left')
    start = rospy.get_rostime()
    test.baxter_ik_move(test.limb,test.pose)
    end = rospy.get_rostime()
    spend_time = end - start
    print spend_time
    test.canny_it()

    print("done")

if __name__ == "__main__":
    main()
