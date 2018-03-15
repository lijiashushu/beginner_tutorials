#!/usr/bin/env python
#coding=utf-8
#####################################################################################
#                                                                                   #
# Copyright (c) 2014, Active Robots Ltd.                                            #
# All rights reserved.                                                              #
#                                                                                   #
# Redistribution and use in source and binary forms, with or without                #
# modification, are permitted provided that the following conditions are met:       #
#                                                                                   #
# 1. Redistributions of source code must retain the above copyright notice,         #
#    this list of conditions and the following disclaimer.                          #
# 2. Redistributions in binary form must reproduce the above copyright              #
#    notice, this list of conditions and the following disclaimer in the            #
#    documentation and/or other materials provided with the distribution.           #
# 3. Neither the name of the Active Robots nor the names of its contributors        #
#    may be used to endorse or promote products derived from this software          #
#    without specific prior written permission.                                     #
#                                                                                   #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"       #
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE         #
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE        #
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE          #
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR               #
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF              #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS          #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN           #
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)           #
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE        #
# POSSIBILITY OF SUCH DAMAGE.                                                       #
#                                                                                   #
#####################################################################################

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

# load the package manifest
roslib.load_manifest("activerobots")

# initialise ros node
rospy.init_node("pick_and_place", anonymous = True)

# directory used to save analysis images
image_directory = os.getenv("HOME") + "/Golf/"

# locate class
class locate():
    def __init__(self, arm, distance):
        global image_directory
        # arm ("left" or "right")
        self.limb           = arm
        self.limb_interface = baxter_interface.Limb(self.limb)

        if arm == "left":
            self.other_limb = "right"
        else:
            self.other_limb = "left"

        self.other_limb_interface = baxter_interface.Limb(self.other_limb)

        # gripper ("left" or "right")
        self.gripper = baxter_interface.Gripper(arm)

        # image directory
        self.image_dir = image_directory

        # flag to control saving of analysis images 如果需要保存的话可以保存，自己来设置
        self.save_images = False

        # required position accuracy in metres
        self.ball_tolerance = 0.010
        self.tray_tolerance = 0.05

        # number of balls found
        self.balls_found = 0

        # start positions  关于球的xyz和关于托盘的xyz
        self.ball_tray_x = 0.6                        # x     = front back
        self.ball_tray_y = 0.49                       # y     = left right
        self.ball_tray_z = 0.155                        # z     = up down 0.15
        self.golf_ball_x = 0.6                        # x     = front back
        self.golf_ball_y = 0                        # y     = left right
        self.golf_ball_z = 0.155                        # z     = up down
        self.roll        = -1.0 * math.pi              # roll  = horizontal  -pi
        self.pitch       = 0.0 * math.pi               # pitch = vertical
        self.yaw         = 0.0 * math.pi               # yaw   = rotation

        self.pose = [self.golf_ball_x, self.golf_ball_y, self.golf_ball_z,\
                     self.roll, self.pitch, self.yaw]  #来回移动，先设置到球的

        # camera parameters (NB. other parameters in open_camera)
        self.cam_calib    = 0.0025                     # meters per pixel at 1 meter
        self.cam_x_offset = 0.045 #0.045                      # camera gripper offset    摄像头中gripper的位置
        self.cam_y_offset = -0.02 #-0.01
        self.width        = 960                        # Camera resolution
        self.height       = 600

        # Hough circle accumulator threshold and minimum radius.  #找圆算法
        self.hough_canny       = 200
        self.hough_accumulator = 20
        self.hough_min_radius  = 20
        self.hough_max_radius  = 70

        # canny image  #用来判断边界的图
        self.canny = cv.CreateImage((self.width, self.height), 8, 1)

        # Canny transform parameters
        self.canny_low  = 60
        self.canny_high = 200

        # minimum ball tray area  像素个数
        self.min_area = 20000

        # callback image
        self.cv_image = cv.CreateImage((self.width, self.height), 8, 3)


        # colours
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)

        # ball tray corners
        self.ball_tray_corner = [(0.0,0.0),(0.0,0.0),(0.0,0.0),(0.0,0.0)]

        # ball tray places
        self.ball_tray_place = [(0.0,0.0),(0.0,0.0),(0.0,0.0),(0.0,0.0),
                                (0.0,0.0),(0.0,0.0),(0.0,0.0),(0.0,0.0),
                                (0.0,0.0),(0.0,0.0),(0.0,0.0),(0.0,0.0)]

        # Enable the actuators
        baxter_interface.RobotEnable().enable()

        # set speed as a ratio of maximum speed
        self.limb_interface.set_joint_position_speed(0.5)
        self.other_limb_interface.set_joint_position_speed(0.5)

        # create image publisher to head monitor   必须制定queue_size
        self.pub = rospy.Publisher('/robot/xdisplay', Image, latch=True, queue_size=10)

        # calibrate the gripper
        self.gripper.calibrate()

        # display the start splash screen
        #self.splash_screen("抓取水果", "放置水果") #在屏幕上显示
        img_for_display = cv2.imread(image_directory+"123.jpg")
        msg = cv_bridge.CvBridge().cv2_to_imgmsg(img_for_display, encoding="bgr8")
        pub = rospy.Publisher('/robot/xdisplay', Image, latch=True)
        pub.publish(msg)




        print("camera setting...")
        # reset cameras
        #########################################################################
        '''横线里面的机器人的摄像头打开后可以注释掉'''
        '''机器人开机运行后显示camere setting done!即打开'''
        # self.reset_cameras()
        #
        # # close all cameras
        # self.close_camera("left")
        # self.close_camera("right")
        # self.close_camera("head")
        #
        # # open required camera
        #
        # self.open_camera(self.limb, self.width, self.height)
        #########################################################################





        # subscribe to required camera
        self.subscribe_to_camera(self.limb)
        print("camere setting done!")

        # distance of arm to table and ball tray
        self.distance      = distance
        self.tray_distance = distance - 0.075

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

    # convert Baxter point to image pixel
    def baxter_to_pixel(self, pt, dist):
        x = (self.width / 2)                                                         \
          + int((pt[1] - (self.pose[1] + self.cam_y_offset)) / (self.cam_calib * dist))
        y = (self.height / 2)                                                        \
          + int((pt[0] - (self.pose[0] + self.cam_x_offset)) / (self.cam_calib * dist))

        return (x, y)

    # convert image pixel to Baxter point
    def pixel_to_baxter(self, px, dist):#将图像上的坐标转换到gripper的坐标
        x = ((px[1] - (self.height / 2)) * self.cam_calib * dist)                \
          + self.pose[0] + self.cam_x_offset
        y = ((px[0] - (self.width / 2)) * self.cam_calib * dist)                 \
          + self.pose[1] + self.cam_y_offset

        return (x, y)

    # Not a tree walk due to python recursion limit  递归限制
    def tree_walk(self, image, x_in, y_in):
        almost_black = (1, 1, 1)

        pixel_list = [(x_in, y_in)]                   # first pixel is black save position
        cv.Set2D(image, y_in, x_in, almost_black)     # set pixel to almost black
        to_do = [(x_in, y_in - 1)]                    # add neighbours to to do list
        to_do.append([x_in, y_in + 1])
        to_do.append([x_in - 1, y_in])
        to_do.append([x_in + 1, y_in])

        while len(to_do) > 0:
            x, y = to_do.pop()                             # get next pixel to test
            if cv.Get2D(image, y, x)[0] == self.black[0]:  # if black pixel found
                pixel_list.append([x, y])                  # save pixel position
                cv.Set2D(image, y, x, almost_black)        # set pixel to almost black
                to_do.append([x, y - 1])                   # add neighbours to to do list
                to_do.append([x, y + 1])
                to_do.append([x - 1, y])
                to_do.append([x + 1, y])
        ##返回全部设为almost black的像素点
        return pixel_list

    # Remove artifacts and find largest object
    # return centre of the ball_tray
    # 图片格式显示时应该反过来
    def look_for_ball_tray(self, canny):

        width, height = cv.GetSize(canny)

        centre   = (0, 0)
        max_area = 0

        # for all but edge pixels
        #从0开始
        for x in range(1, width - 2):
            for y in range(1, height - 2):
                if cv.Get2D(canny, y, x)[0] == self.black[0]:       # black pixel found
                    pixel_list = self.tree_walk(canny, x, y)        # tree walk pixel
                    if len(pixel_list) < self.min_area:             # if object too small
                        for l in pixel_list:
                            cv.Set2D(canny, l[1], l[0], self.white) # set pixel to white
                    else:                                           # if object found
                        n = len(pixel_list)
                        if n > max_area:                            # if largest object found
                            sum_x  = 0                              # find centre of object
                            sum_y  = 0
                            for p in pixel_list:
                                sum_x  = sum_x + p[0]
                                sum_y  = sum_y + p[1]

                            centre = sum_x / n, sum_y / n           # save centre of object
                            max_area = n                            # save area of object
                            #像素的个数
        if max_area > 0:                                            # in tray found
            cv.Circle(canny, (centre), 9, (250, 250, 250), -1)      # mark tray centre

        # display the modified canny
        #cv.ShowImage("Modified Canny", canny)

        # 3ms wait
        cv.WaitKey(3)

        return centre                                        # return centre of object

    # flood fill edge of image to leave objects
    def flood_fill_edge(self, canny):
        width, height = cv.GetSize(canny)

        # set boarder pixels to white  把边界全弄白
        for x in range(width):
            cv.Set2D(canny, 0, x, self.white)
            cv.Set2D(canny, height - 1, x, self.white)

        for y in range(height):
            cv.Set2D(canny, y, 0, self.white)
            cv.Set2D(canny, y, width - 1, self.white)

        # prime to do list   把这些地方变白（边界）
        to_do = [(2, 2)]
        to_do.append([2, height - 3])
        to_do.append([width - 3, height - 3])
        to_do.append([width - 3, 2])

        #遍历所有添加到List中，什么时候停止？
        while len(to_do) > 0:
            x, y = to_do.pop()                               # get next pixel to test
            if cv.Get2D(canny, y, x)[0] == self.black[0]:    # if black pixel found
                cv.Set2D(canny, y, x, self.white)            # set pixel to white
                to_do.append([x, y - 1])                     # add neighbours to to do list
                to_do.append([x, y + 1])
                to_do.append([x - 1, y])
                to_do.append([x + 1, y])



        # display the flood
        #cv.ShowImage("flood Canny", canny)
        file_name = self.image_dir + "flood.jpg"
        cv.SaveImage(file_name, canny)

        # 3ms wait
        cv.WaitKey(3)

    # camera call back function
    def camera_callback(self, data, camera_name):
        # Convert image from a ROS image message to a CV image
        try:
            self.cv_image = cv_bridge.CvBridge().imgmsg_to_cv2(data, "bgr8")
        except cv_bridge.CvBridgeError, e:
            print e

        # 3ms wait
        cv.WaitKey(3)

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

        #订阅摄像头发布的消息
        camera_sub = rospy.Subscriber(camera_str, Image, callback)

    # Convert cv image to a numpy array  转化成numpy数组
    def cv2array(self, im):
        depth2dtype = {cv.IPL_DEPTH_8U: 'uint8',
                       cv.IPL_DEPTH_8S: 'int8',
                       cv.IPL_DEPTH_16U: 'uint16',
                       cv.IPL_DEPTH_16S: 'int16',
                       cv.IPL_DEPTH_32S: 'int32',
                       cv.IPL_DEPTH_32F: 'float32',
                       cv.IPL_DEPTH_64F: 'float64'}
  
        arrdtype=im.depth
        a = numpy.fromstring(im.tostring(),
                             dtype = depth2dtype[im.depth],
                             count = im.width * im.height * im.nChannels)
        a.shape = (im.height, im.width, im.nChannels)

        return a

    # find next object of interest
    def find_next_golf_ball(self, ball_data, iteration):
        # if only one object then object found
        if len(ball_data) == 1:
            return ball_data[0]

        # sort objects right to left
        od = []
        for i in range(len(ball_data)):
            od.append(ball_data[i])

        od.sort()
        return od[0]
        '''
        # if one ball is significantly to the right of the others
        if od[1][0] - od[0][0] > 30:       # if ball significantly to right of the others
            return od[0]                   # return right most ball
        elif od[1][1] < od[0][1]:          # if right most ball below second ball
            return od[0]                   # return lower ball
        else:                              # if second ball below right most ball
            return od[1]                   # return lower ball
        '''
    # find gripper angle to avoid nearest neighbour
    def find_gripper_angle(self, next_ball, ball_data):
        # if only one ball any angle will do
        if len(ball_data) == 1:
            return self.yaw

        # find nearest neighbour
        neighbour = (0, 0)
        min_d2    = float(self.width * self.width + self.height * self.height)
        #距离的平方

        #所有的球除了自己，找到距离最小的球
        for i in range(len(ball_data)):
            if ball_data[i][0] != next_ball[0] or ball_data[i][1] != next_ball[1]:
                dx = float(ball_data[i][0]) - float(next_ball[0])   # NB x and y are ushort
                dy = float(ball_data[i][1]) - float(next_ball[1])   # float avoids error
                d2 = (dx * dx) + (dy * dy)
                if d2 < min_d2:
                    neighbour = ball_data[i]
                    min_d2    = d2

        # find best angle to avoid hitting neighbour
        dx = float(next_ball[0]) - float(neighbour[0])
        dy = float(next_ball[1]) - float(neighbour[1])
        if abs(dx) < 10.0:
            angle = - (math.pi / 2.0)             # avoid divide by zero
            print("aaaaa")
            return 0
        else:
            angle = math.atan(dy / dx)            # angle in radians between -pi and pi
            print ("\n")
            print ("\n")
            print ("dy:")
            print (dy)
            print ("dx:")
            print (dx)
            print ("angle")
            print (angle)
            print ("\n")
            if angle > 0:
                print("bbbbb")
                return  math.pi/2-angle
            else:
                print("ccccc")
                return   -(angle+math.pi/2)

        """
        angle = angle + (math.pi / 2.0)           # rotate pi / 2 radians
        if angle > (math.pi / 2.0):                 # ensure angle between -pi and pi
            angle = angle - math.pi
        """



                            # return best angle to grip golf ball

    # if ball near any of the ball tray places
    def is_near_ball_tray(self, ball):
        for i in self.ball_tray_place:
            d2 = ((i[0] - ball[0]) * (i[0] - ball[0]))           \
               + ((i[1] - ball[1]) * (i[1] - ball[1]))
            if d2 < 0.0004:
               return True

        return False

    # Use Hough circles to find ball centres (Only works with round objects)
    # 找到图片中的ball，并且返回一个x，y坐标和抓取的角度
    # return next_ball, angle
    def hough_it(self, n_ball, iteration):

        # create gray scale image of balls
        gray_image = cv.CreateImage((self.width, self.height), 8, 1)

        trans = cv.fromarray(self.cv_image, )

        #（src，dst，模式：其中当code选用CV_BGR2GRAY时，dst需要是单通道图片）
        cv.CvtColor(trans, gray_image, cv.CV_BGR2GRAY)

        # create gray scale array of balls
        gray_array = self.cv2array(gray_image)

        # find Hough circles
        # gray 输入 8-比特、单通道灰度图像.
        # param1: canny threshold
        # param2: The smaller it is, the more false circles may be detected.
        # 输出为3个参数组成的空间，x，y，r
        circles = cv2.HoughCircles(gray_array, cv.CV_HOUGH_GRADIENT, 1, 40, param1=self.hough_canny,  \
                  param2=self.hough_accumulator, minRadius=self.hough_min_radius,       \
                  maxRadius=self.hough_max_radius)

        # Check for at least one ball found
        if circles is None:
            # display no balls found message on head display
            #self.splash_screen("no balls", "found")
            # no point in continuing so exit with error message
            sys.exit("ERROR - hough_it - No golf balls found")

        #around 小数近似为整数
        #uint16 转化为uint16的array
        circles = numpy.uint16(numpy.around(circles))

        ball_data = {}
        n_balls   = 0

        #转化为array
        circle_array = numpy.asarray(self.cv_image)

        # check if golf ball is in ball tray
        # 并且找出有几个ball，保存在n_ball 中
        for i in circles[0,:]: #第0行的所有列，应该一共有三行
            # convert to baxter coordinates
            #(x,y)
            ball = self.pixel_to_baxter((i[0], i[1]), self.tray_distance)

            if self.is_near_ball_tray(ball):
                # draw the outer circle in red
                # 图片名称、圆心坐标、半径、颜色数组、线宽
                cv2.circle(circle_array, (i[0], i[1]), i[2], (0, 0, 255), 2)
                # draw the center of the circle in red
                cv2.circle(circle_array, (i[0], i[1]), 2, (0, 0, 255), 3)
            elif i[1] > 800:
                # draw the outer circle in red
                cv2.circle(circle_array, (i[0], i[1]), i[2], (0, 0, 255), 2)
                # draw the center of the circle in red
                cv2.circle(circle_array, (i[0], i[1]), 2, (0, 0, 255), 3)
            else:#确定是要抓的ball
                # draw the outer circle in green
                cv2.circle(circle_array, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle in green
                cv2.circle(circle_array, (i[0], i[1]), 2, (0, 255, 0), 3)

                ball_data[n_balls]  = (i[0], i[1], i[2])
                n_balls            += 1

        #把numpy的narray转化为opencv image
        circle_image = cv.fromarray(circle_array)

        cv.ShowImage("Hough Circle", circle_image)

        # 3ms wait
        cv.WaitKey(3)

        # display image on head monitor
        font     = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 1)
        position = (30, 60)
        s = "Searching for golf balls"
        #cv.PutText(circle_image, s, position, font, self.white)
        msg = cv_bridge.CvBridge().cv2_to_imgmsg(circle_array, encoding="bgr8")
        self.pub.publish(msg)


        # save image of Hough circles on raw image
        file_name = self.image_dir                                                 \
                  + "hough_circle_" + str(n_ball) + "_" + str(iteration) + ".jpg"
        cv.SaveImage(file_name, circle_image)

        # Check for at least one ball found
        if n_balls == 0:                    # no balls found
            # display no balls found message on head display
            #self.splash_screen("no balls", "found")
            # less than 12 balls found, no point in continuing, exit with error message
            sys.exit("ERROR - hough_it - No golf balls found")

        # select next ball and find it's position
        next_ball = self.find_next_golf_ball(ball_data, iteration)

        # find best gripper angle to avoid touching neighbouring ball
        angle = self.find_gripper_angle(next_ball, ball_data)

        # return next golf ball position and pickup angle
        return next_ball, angle

    # move a limb
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

    # find distance of limb from nearest line of sight object
    def get_distance(self, limb):
        if limb == "left":
            dist = baxter_interface.analog_io.AnalogIO('left_hand_range').state()
        elif limb == "right":
            dist = baxter_interface.analog_io.AnalogIO('right_hand_range').state()
        else:
            sys.exit("ERROR - get_distance - Invalid limb")

        # convert mm to m and return distance
        return float(dist / 1000.0)

    # update pose in x and y direction
    def update_pose(self, dx, dy):#移动到新的位置
        x = self.pose[0] + dx
        y = self.pose[1] + dy
        pose = [x, y, self.pose[2], self.roll, self.pitch, self.yaw]
        self.baxter_ik_move(self.limb, pose)

    # used to place camera over the ball tray
    #不断计算tray的中心并且移动直到中心位于摄像头得到图像的中心
    #没有直接用到图像
    def ball_tray_iterate(self, iteration, centre):
        # print iteration number
        print "Egg Tray Iteration ", iteration

        # find displacement of object from centre of image
        pixel_dx    = (self.width / 2) - centre[0]
        pixel_dy    = (self.height / 2) - centre[1]
        pixel_error = math.sqrt((pixel_dx * pixel_dx) + (pixel_dy * pixel_dy))
        error       = float(pixel_error * self.cam_calib * self.tray_distance)

        x_offset = - pixel_dy * self.cam_calib * self.tray_distance
        y_offset = - pixel_dx * self.cam_calib * self.tray_distance

        # if error in current position too big
        if error > self.tray_tolerance:
            # correct pose
            self.update_pose(x_offset, y_offset)
            # find new centre
            centre = self.canny_it(iteration)

            # find displacement of object from centre of image
            pixel_dx    = (self.width / 2) - centre[0]
            pixel_dy    = (self.height / 2) - centre[1]
            pixel_error = math.sqrt((pixel_dx * pixel_dx) + (pixel_dy * pixel_dy))
            error       = float(pixel_error * self.cam_calib * self.tray_distance)
            #计算如果满足的话需要直接停止迭代，所以需要上下都算

        return centre, error

    # randomly adjust a pose to dither arm position
    # used to prevent stalemate when looking for ball tray
    def dither(self):
        x = self.ball_tray_x
        y = self.ball_tray_y + (random.random() / 10.0)
        pose = (x, y, self.ball_tray_z, self.roll, self.pitch, self.yaw)

        return pose

    # find the ball tray
    # return ball_tray_centre
    # 改了图片格式转化
    def canny_it(self, iteration):
        ####图片格式转化#################################
        cv_image_trans = cv.fromarray(self.cv_image)

        #保存self.cv_image
        if self.save_images:
            # save raw image of ball tray
            file_name = self.image_dir + "ball_tray_" + str(iteration) + ".jpg"
            cv.SaveImage(file_name, self.cv_image)

        # create an empty image variable, the same dimensions as our camera feed.
        gray = cv.CreateImage((cv.GetSize(cv_image_trans)), 8, 1)

        # convert the image to a grayscale image
        #(src,dst,code)

        cv.CvtColor(cv_image_trans, gray, cv.CV_BGR2GRAY)

        # display image on head monitor
        font     = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 1)
        position = (30, 60)
        #cv.PutText(cv_image_trans, "Looking for ball tray", position, font, self.white)
        msg = cv_bridge.CvBridge().cv2_to_imgmsg(self.cv_image, encoding="bgr8")
        self.pub.publish(msg)

        # create a canny edge detection map of the greyscale image
        cv.Canny(gray, self.canny, self.canny_low, self.canny_high, 3)

        # display the canny transformation
        cv.ShowImage("Canny Edge Detection", self.canny)

        #保存self.canny
        if self.save_images:
            # save Canny image of ball tray
            file_name = self.image_dir + "canny_tray_" + str(iteration) + ".jpg"
            cv.SaveImage(file_name, self.canny)

        # flood fill edge of image to leave only objects
        self.flood_fill_edge(self.canny)
        ball_tray_centre = self.look_for_ball_tray(self.canny)#可能会返回（0,0）

        # 3ms wait
        cv.WaitKey(3)

        #第一次进来还未找到时起作用
        while ball_tray_centre[0] == 0:
            if random.random() > 0.6:
                self.baxter_ik_move(self.limb, self.dither())

            ball_tray_centre = self.canny_it(iteration)

        return ball_tray_centre

    # find places for golf balls 放球的位置
    #改了格式转化
    def find_places(self, c):
        ####图片格式转化#################################
        cv_image_trans = cv.fromarray(self.cv_image)
        # find long side of ball tray
        l1_sq = ((c[1][0] - c[0][0]) * (c[1][0] - c[0][0])) +           \
                ((c[1][1] - c[0][1]) * (c[1][1] - c[0][1]))
        l2_sq = ((c[2][0] - c[1][0]) * (c[2][0] - c[1][0])) +           \
                ((c[2][1] - c[1][1]) * (c[2][1] - c[1][1]))

        if l1_sq > l2_sq:                     # c[0] to c[1] is a long side
            cc = [c[0], c[1], c[2], c[3]]
        else:                                 # c[1] to c[2] is a long side
            cc = [c[1], c[2], c[3], c[0]]

        # ball tray corners in baxter coordinates
        for i in range(4):
            self.ball_tray_corner[i] = self.pixel_to_baxter(cc[i], self.tray_distance)

        # ball tray places in pixel coordinates
        ref_x = cc[0][0]
        ref_y = cc[0][1]
        dl_x  = (cc[1][0] - cc[0][0]) / 8
        dl_y  = (cc[1][1] - cc[0][1]) / 8
        ds_x  = (cc[2][0] - cc[1][0]) / 6
        ds_y  = (cc[2][1] - cc[1][1]) / 6

        #算出放置球的12位置
        p     = {}

        p[0] = (ref_x + (1 * dl_x) + (1 * ds_x), ref_y + (1 * dl_y) + (1 * ds_y))
        p[1]  = (ref_x + (1 * dl_x) + (3 * ds_x), ref_y + (1 * dl_y) + (3 * ds_y))
        p[2] = (ref_x + (1 * dl_x) + (5 * ds_x), ref_y + (1 * dl_y) + (5 * ds_y))
        p[3] = (ref_x + (3 * dl_x) + (1 * ds_x), ref_y + (3 * dl_y) + (1 * ds_y))
        p[4] = (ref_x + (3 * dl_x) + (3 * ds_x), ref_y + (3 * dl_y) + (3 * ds_y))
        p[5] = (ref_x + (3 * dl_x) + (5 * ds_x), ref_y + (3 * dl_y) + (5 * ds_y))
        p[6] = (ref_x + (5 * dl_x) + (1 * ds_x), ref_y + (5 * dl_y) + (1 * ds_y))
        p[7] = (ref_x + (5 * dl_x) + (3 * ds_x), ref_y + (5 * dl_y) + (3 * ds_y))
        p[8] = (ref_x + (5 * dl_x) + (5 * ds_x), ref_y + (5 * dl_y) + (5 * ds_y))
        p[9]  = (ref_x + (7 * dl_x) + (1 * ds_x), ref_y + (7 * dl_y) + (1 * ds_y))
        p[10] = (ref_x + (7 * dl_x) + (3 * ds_x), ref_y + (7 * dl_y) + (3 * ds_y))
        p[11] = (ref_x + (7 * dl_x) + (5 * ds_x), ref_y + (7 * dl_y) + (5 * ds_y))

        #在这些位置上画圈

        for i in range(12):
            # mark position of ball tray places
            cv.Circle(cv_image_trans, (int(p[i][0]), int(p[i][1])), 5, (0, 250, 0), -1)

            #位置标序号
            font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 1)
            cv.PutText(cv_image_trans, str(i+1),(int(p[i][0]), int(p[i][1])), font, (0, 255, 0))
            # ball tray places in baxter coordinates
            # 把这些位置转换到baxter的坐标系
            self.ball_tray_place[i] = self.pixel_to_baxter(p[i], self.tray_distance)

        # display the ball tray places
        file_name = self.image_dir + "eggtray.jpg"
        cv.SaveImage(file_name, cv_image_trans)
        cv.ShowImage("Egg tray", cv_image_trans)

        if self.save_images:
            # save ball tray image with overlay of ball tray and ball positions
            file_name = self.image_dir + "ball_tray.jpg"
            cv.SaveImage(file_name, cv_image_trans)

        # 3ms wait
        cv.WaitKey(3)

    # find four corners of the ball tray
    # 改了图片格式转化
    def find_corners(self, centre):
        ####图片格式转化#################################
        cv_image_trans = cv.fromarray(self.cv_image)
        # find bottom corner#######################################################
        max_x  = 0
        max_y  = 0
        #默认在屏幕长左右各减100 宽上下各减20的范围内  从下向上，从左到右
        for x in range(100, self.width - 100):
            y = self.height - 20      #580
            while y > 0 and cv.Get2D(self.canny, y, x)[0] > 100:
                y = y - 1
            if y > 20:

                cv.Set2D(cv_image_trans, y, x, (0, 0, 255))
                if y > max_y:
                    max_x = x
                    max_y = y

        corner_1 = (max_x, max_y)

        # find left corner#######################################################
        min_x  = self.width
        min_y  = 0

        # 思想相同，默认在屏幕长左右各减100 宽上下各减20的范围内  从下向上，从左到右
        for y in range(100, self.height - 100):
            x = 20
            while x < self.width - 1 and cv.Get2D(self.canny, y, x)[0] > 100:
                x = x + 1
            if x < self.width - 20:
                cv.Set2D(cv_image_trans, y, x, (0, 255, 0, 0))
                if x < min_x:
                    min_x = x
                    min_y = y

        corner_2 = (min_x, min_y)

        # display corner image#######################################################
        file_name = self.image_dir + "corner.jpg"
        cv.SaveImage(file_name, cv_image_trans)
        cv.ShowImage("Corner", cv_image_trans)

        if self.save_images:
            # save Canny image
            file_name = self.image_dir + "egg_tray_canny.jpg"
            cv.SaveImage(file_name, self.canny)

            # mark corners and save corner image
            cv.Circle(self.cv_image, corner_1, 9, (0, 250, 0), -1)
            cv.Circle(self.cv_image, corner_2, 9, (0, 250, 0), -1)
            file_name = self.image_dir + "corner.jpg"
            cv.SaveImage(file_name, self.cv_image)

        # 3ms wait
        cv.WaitKey(3)

        # two corners found and centre known find other two corners#####################
        corner_3 = ((2 * centre[0]) - corner_1[0], (2 * centre[1]) - corner_1[1])
        corner_4 = ((2 * centre[0]) - corner_2[0], (2 * centre[1]) - corner_2[1])

        # draw ball tray boundry
        c1 = (int(corner_1[0]), int(corner_1[1]))
        c2 = (int(corner_2[0]), int(corner_2[1]))
        c3 = (int(corner_3[0]), int(corner_3[1]))
        c4 = (int(corner_4[0]), int(corner_4[1]))

        cv.Line(cv_image_trans, c1, c2, (255, 0, 0), thickness=3)
        cv.Line(cv_image_trans, c2, c3, (255, 0, 0), thickness=3)
        cv.Line(cv_image_trans, c3, c4, (255, 0, 0), thickness=3)
        cv.Line(cv_image_trans, c4, c1, (255, 0, 0), thickness=3)

        return True, (corner_1, corner_2, corner_3, corner_4)

    # find the ball tray
    # 没有用到图片格式转化
    def find_ball_tray(self):
        ok = False
        while not ok:
            ball_tray_centre = self.canny_it(0)

            error     = 2 * self.tray_tolerance #先满足循环的条件
            iteration = 1 #第一次迭代次数计数

            # iterate until arm over centre of tray
            while error > self.tray_tolerance:
                ball_tray_centre, error = self.ball_tray_iterate(iteration, ball_tray_centre)
                iteration              += 1
            #移动到中心之后寻找Corners
            # find ball tray corners in pixel units
            (ok, corners) = self.find_corners(ball_tray_centre)

        self.find_places(corners)

    # used to place camera over golf ball
    def golf_ball_iterate(self, n_ball, iteration, ball_data):
        # print iteration number
        print "GOLF BALL", n_ball, "ITERATION ", iteration

        # find displacement of ball from centre of image
        pixel_dx    = (self.width / 2) - ball_data[0]
        pixel_dy    = (self.height / 2) - ball_data[1]
        pixel_error = math.sqrt((pixel_dx * pixel_dx) + (pixel_dy * pixel_dy))
        error       = float(pixel_error * self.cam_calib * self.tray_distance)

        x_offset = - pixel_dy * self.cam_calib * self.tray_distance
        y_offset = - pixel_dx * self.cam_calib * self.tray_distance

        # update pose and find new ball data
        self.update_pose(x_offset, y_offset)
        ball_data, angle = self.hough_it(n_ball, iteration)

        # find displacement of ball from centre of image
        pixel_dx    = (self.width / 2) - ball_data[0]
        pixel_dy    = (self.height / 2) - ball_data[1]
        pixel_error = math.sqrt((pixel_dx * pixel_dx) + (pixel_dy * pixel_dy))
        error       = float(pixel_error * self.cam_calib * self.tray_distance)

        return ball_data, angle, error

    # print all 6 arm coordinates (only required for programme development)
    def print_arm_pose(self):
        return
        pi = math.pi

        quaternion_pose = self.limb_interface.endpoint_pose()
        position        = quaternion_pose['position']
        quaternion      = quaternion_pose['orientation']
        euler           = tf.transformations.euler_from_quaternion(quaternion)

        print
        print "             %s" % self.limb
        print 'front back = %5.4f ' % position[0]
        print 'left right = %5.4f ' % position[1]
        print 'up down    = %5.4f ' % position[2]
        print 'roll       = %5.4f radians %5.4f degrees' %euler[0], 180.0 * euler[0] / pi
        print 'pitch      = %5.4f radians %5.4f degrees' %euler[1], 180.0 * euler[1] / pi
        print 'yaw        = %5.4f radians %5.4f degrees' %euler[2], 180.0 * euler[2] / pi

    # find all the golf balls and place them in the ball tray
    def pick_and_place(self):
        ####图片格式转化#################################
        cv_image_trans = cv.fromarray(self.cv_image)
        n_ball = 0
        while True and n_ball <3:              # assume no more than 12 golf balls
            n_ball          += 1
            iteration        = 0
            angle            = 0.0

            # use Hough circles to find balls and select one ball
            next_ball, angle = self.hough_it(n_ball, iteration)

            error     = 2 * self.ball_tolerance

            print
            print "Ball number ", n_ball
            print "==============="

            # iterate to find next golf ball
            # if hunting to and fro accept error in position
            while error > self.ball_tolerance and iteration < 10:
                iteration               += 1
                next_ball, angle, error  = self.golf_ball_iterate(n_ball, iteration, next_ball)
			
            ####monitor display###########################################
            font     = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 1)
            position = (30, 60)
            s        = "Picking up golf ball"
            #cv.PutText(cv_image_trans, s, position, font, self.white)
            msg = cv_bridge.CvBridge().cv2_to_imgmsg(self.cv_image, encoding="bgr8")
            self.pub.publish(msg)
            ##############################################################

            print "DROPPING BALL ANGLE =", angle * (math.pi / 180)
            if angle != self.yaw:             # if neighbouring ball
                pose = (self.pose[0],         # rotate gripper to avoid hitting neighbour
                        self.pose[1],
                        self.pose[2],
                        self.pose[3],
                        self.pose[4],
                        angle)
                self.baxter_ik_move(self.limb, pose)##rotate

                #cv.PutText(cv_image_trans, s, position, font, self.white)
                msg = cv_bridge.CvBridge().cv2_to_imgmsg(self.cv_image, encoding="bgr8")
                self.pub.publish(msg)

            # slow down to reduce scattering of neighbouring golf balls
            self.limb_interface.set_joint_position_speed(0.2)

            # move down to pick up ball
            pose = (self.pose[0] + self.cam_x_offset,
                    self.pose[1] + self.cam_y_offset,
                    self.pose[2] + (0 - self.distance),  #0.112
                    self.pose[3],
                    self.pose[4],
                    angle)
            self.baxter_ik_move(self.limb, pose)
            self.print_arm_pose()#

            # close the gripper
            self.gripper.close()

            # move down up wiht ball
            pose = (self.pose[0] + self.cam_x_offset,
                    self.pose[1] + self.cam_y_offset,
                    self.pose[2] ,  # 0.112
                    self.pose[3],
                    self.pose[4],
                    angle)
            self.baxter_ik_move(self.limb, pose)

            self.limb_interface.set_joint_position_speed(0.5)

            s = "Moving to ball tray"
            #cv.PutText(cv_image_trans, s, position, font, self.white)
            msg = cv_bridge.CvBridge().cv2_to_imgmsg(self.cv_image, encoding="bgr8")
            self.pub.publish(msg)

            if n_ball == 1:
                err = -0.02
            elif n_ball==2:
                err = 0.02
            elif n_ball == 3:
                err = 0.04
            else:
                err = 0

            pose = (self.ball_tray_place[n_ball - 1][0]+err,  # 提前保存了每个位置的坐标
                    self.ball_tray_place[n_ball - 1][1]-0.03,
                    self.pose[2],
                    self.pose[3],
                    self.pose[4],
                    self.pose[5])
            self.baxter_ik_move(self.limb, pose)

            # display current image on head display
            #cv.PutText(cv_image_trans, s, position, font, self.white)
            msg = cv_bridge.CvBridge().cv2_to_imgmsg(self.cv_image, encoding="bgr8")
            self.pub.publish(msg)


            self.limb_interface.set_joint_position_speed(0.2)
            # move down
            pose = (self.ball_tray_place[n_ball - 1][0]+err,   #提前保存了每个位置的坐标
                    self.ball_tray_place[n_ball - 1][1]-0.03,
                    self.pose[2] + 0.04 - self.distance,
                    self.pose[3],
                    self.pose[4],
                    self.pose[5])
            self.baxter_ik_move(self.limb, pose)

            # display current image on head display
            s = "Placing golf ball in ball tray"
            #cv.PutText(cv_image_trans, s, position, font, self.white)
            msg = cv_bridge.CvBridge().cv2_to_imgmsg(self.cv_image, encoding="bgr8")
            self.pub.publish(msg)

            # open the gripper
            self.gripper.open()

            # move up
            pose = (self.ball_tray_place[n_ball - 1][0],  # 提前保存了每个位置的坐标
                    self.ball_tray_place[n_ball - 1][1],
                    self.pose[2],
                    self.pose[3],
                    self.pose[4],
                    self.pose[5])
            self.baxter_ik_move(self.limb, pose)

            self.limb_interface.set_joint_position_speed(0.5)
            # prepare to look for next ball     #移动到找球的位置
            self.limb_interface.set_joint_position_speed(0.5)
            pose = (self.golf_ball_x,
                    self.golf_ball_y,
                    self.golf_ball_z,
                    -1.0 * math.pi,
                    0.0 * math.pi,
                    0.0 * math.pi)
            self.baxter_ik_move(self.limb, pose)

        # display all balls found on head display
        #self.splash_screen("all balls", "found")

        print "All balls found"

    # display message on head display
    def splash_screen(self, s1, s2):
        splash_array = numpy.zeros((self.height, self.width, 3), numpy.uint8)
        font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 3.0, 3.0, 9)

        ((text_x, text_y), baseline) = cv.GetTextSize(s1, font)
        org = ((self.width - text_x) / 2, (self.height / 3) + (text_y / 2))
        cv2.putText(splash_array, s1, org, cv.CV_FONT_HERSHEY_SIMPLEX, 3.0,          \
                    self.white, thickness = 7)

        ((text_x, text_y), baseline) = cv.GetTextSize(s2, font)
        org = ((self.width - text_x) / 2, ((2 * self.height) / 3) + (text_y / 2))
        cv2.putText(splash_array, s2, org, cv.CV_FONT_HERSHEY_SIMPLEX, 3.0,          \
                    self.white, thickness = 7)
        #启动动画
        #fromarray convert between opencv image and numpy ndarray
        splash_image = cv.fromarray(splash_array)


        # 3ms wait
        cv2.waitKey(3)
        #改
        msg = cv_bridge.CvBridge().cv2_to_imgmsg(splash_array, encoding="bgr8")
        #msg = cv_bridge.CvBridge().cv2_to_imgmsg(splash_image, encoding="bgr8")
        self.pub.publish(msg)

# read the setup parameters from setup.dat
def get_setup():
    global image_directory
    file_name = image_directory + "setup.dat"

    try:
        f = open(file_name, "r")
    except ValueError:
        sys.exit("ERROR: golf_setup must be run before golf")

    # find limb
    s = string.split(f.readline())
    if len(s) >= 3:
        if s[2] == "left" or s[2] == "right":
            limb = s[2]
        else:
            sys.exit("ERROR: invalid limb in %s" % file_name)
    else:
        sys.exit("ERROR: missing limb in %s" % file_name)

    # find distance to table
    s = string.split(f.readline())
    if len(s) >= 3:
        try:
            distance = float(s[2])
        except ValueError:
            sys.exit("ERROR: invalid distance in %s" % file_name)
    else:
        sys.exit("ERROR: missing distance in %s" % file_name)

    return limb, distance

def main():
    # get setup parameters
    limb, distance = get_setup()
    print "limb     = ", limb
    print "distance = ", distance

    # create locate class instance
    locator = locate(limb, distance)

    raw_input("Press Enter to start: ")

    # open the gripper
    locator.gripper.open()

    # move close to the ball tray
    locator.pose = (locator.ball_tray_x,
                    locator.ball_tray_y,
                    locator.ball_tray_z,
                    locator.roll,
                    locator.pitch,
                    locator.yaw)
    locator.baxter_ik_move(locator.limb, locator.pose)

    # find the ball tray
    locator.find_ball_tray()

    # find all the golf balls and place them in the ball tray
    locator.pose = (locator.golf_ball_x,
                    locator.golf_ball_y,
                    locator.golf_ball_z,
                    locator.roll,
                    locator.pitch,
                    locator.yaw)
    locator.baxter_ik_move(locator.limb, locator.pose)
    locator.pick_and_place()

if __name__ == "__main__":
    main()

