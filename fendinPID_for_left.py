#!/usr/bin/env python
# coding=utf-8
import math
import numpy as np
import baxter_interface
import sys
import rospy
from moveit_commander import conversions
import matplotlib.pyplot as plt
import threading
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
#from baxter_pykdl import baxter_kinematics
from baxter_core_msgs.msg import SEAJointState
from baxter_core_msgs.msg import JointCommand
from tf import transformations
from beginner_tutorials.srv import dynamics
from beginner_tutorials.srv import dynamicsRequest

import struct
import socket

class PIDController(object):
    def __init__(self):
        self._prev_err = 0.0

        self._kp = 0.0
        self._ki = 0.0
        self._kd = 0.0
        # initialize error, results
        self._cp = 0.0
        self._ci = 0.0
        self._cd = 0.0

        self._cur_time = 0.0
        self._prev_time = 0.0

        self.initialize()

    def initialize(self):
        self._cur_time = rospy.get_time()
        self._prev_time = self._cur_time

        self._prev_err = 0.0

        self._cp = 0.0
        self._ci = 0.0
        self._cd = 0.0

    def set_kp(self, invar):
        self._kp = invar

    def set_ki(self, invar):
        self._ki = invar

    def set_kd(self, invar):
        self._kd = invar

    def get_kp(self):
        return self._kp

    def get_ki(self):
        return self._ki

    def get_kd(self):
        return self._kd

    def compute_output(self, error):
        """
        Performs a PID computation and returns a control value based on
        the elapsed time (dt) and the error signal from a summing junction
        (the error parameter).
        """
        self._cur_time = rospy.get_time()  # get t
        dt = self._cur_time - self._prev_time  # get delta t
        de = error - self._prev_err  # get delta error
        if error <=0.01 and error>=-0.01:
            self._cp = 0  # proportional term
        else:
            self._cp = error
        self._ci += error * dt  # integral term

        self._cd = 0
        if dt > 0:  # no div by zero
            self._cd = de / dt  # derivative term

        self._prev_time = self._cur_time  # save t for next pass
        self._prev_err = error  # save t-1 error

        result = ((self._kp * self._cp) + (self._ki * self._ci) + (self._kd * self._cd))
        return  result

class JointControl(object):
    def __init__(self, ArmName):
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable()
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

        self.torcmd_now = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
        self.name = ArmName
        self.limb = baxter_interface.Limb(ArmName)
        #self.limb.move_to_neutral()

        t1 = self.limb.joint_angles()
        t2 = [0.938, 1.491, 2.167, -1.348, 1.393, -0.466, -0.586]
        temp = 0
        for key in t1:
            t1[key] = t2[temp]
            temp = temp + 1
        self.limb.move_to_joint_positions(t1)
        #self.limb.set_joint_position_speed(0.1)

        self.actual_effort = self.limb.joint_efforts()
        self.gravity_torques = self.limb.joint_efforts()
        self.final_torques = self.gravity_torques.copy()

        self.qnow = dict()  # 得到关节角度的字典
        self.qnow_value = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 得到角度的值
        self.torcmd = dict()  # 给baxter赋值

        self.torController0 = PIDController()
        self.torController1 = PIDController()
        self.torController2 = PIDController()
        self.torController3 = PIDController()
        self.torController4 = PIDController()
        self.torController5 = PIDController()
        self.torController6 = PIDController()

        self.torController = \
            [self.torController0, self.torController1, self.torController2, self.torController3,
             self.torController4, self.torController5, self.torController6
             ]

        '''设置PID参数'''
        # 最前端

        self.torController[0].set_kp(50)  # 130#80.0#*0.6
        self.torController[0].set_ki(0.2)
        self.torController[0].set_kd(2.5)  # 10#15#0.01#*0.6#21.0

        self.torController[1].set_kp(60)  # 130#80.0#*0.6
        self.torController[1].set_ki(0.2)
        self.torController[1].set_kd(1.3)  # 10#15#0.01#*0.6#21.0

        self.torController[2].set_kp(5.1)
        self.torController[2].set_ki(0.1)  # 0.1
        self.torController[2].set_kd(2.5)

        self.torController[3].set_kp(14)  # 130#80.0#*0.6
        self.torController[3].set_ki(0.2)  # 0.05
        self.torController[3].set_kd(3)  # 10#15#0.01#*0.6#21.0

        self.torController[4].set_kp(25)  # 130#80.0#*0.6
        self.torController[4].set_ki(0.2)
        self.torController[4].set_kd(3)  # 10#15#0.01#*0.6#21.0

        self.torController[5].set_kp(12)  # 130#80.0#*0.6
        self.torController[5].set_ki(0)
        self.torController[5].set_kd(10)  # 10#15#0.01#*0.6#21.0

        self.torController[6].set_kp(12)  # 130#80.0#*0.6
        self.torController[6].set_ki(0.1)
        self.torController[6].set_kd(10)  # 10#15#0.01#*0.6#21.0

        # self.subscribe_to_gravity_compensation()

    '''力矩控制'''
    def torquecommand(self, qd):  # qd为期望轨迹
        self.err = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.qnow = self.limb.joint_angles()  # 得到每个关节的当前角度

        temp = 0
        for key in self.qnow:
            self.qnow_value[temp] = self.qnow[key]  # 转化为list
            temp = temp + 1
        # print self.qnow_value
        self.err = qd - self.qnow_value  # 计算每个关节角度的误差
        # print self.err
        self.torcmd = self.limb.joint_efforts()
        # print self.torcmd
        temp = 0
        for key in self.torcmd:
            self.torcmd[key] = self.torController[temp].compute_output(self.err[temp])

            temp = temp + 1


        self.final_torques = self.gravity_torques.copy()  ##########又把final 和gravity连在一起了######
        for key in self.torcmd:
            self.final_torques[key] = self.torcmd[key]
        return self.torcmd

    def gravity_callback(self, data):
        frommsg1 = data.gravity_model_effort
        frommsg2 = data.actual_effort

        temp = 0
        for key in self.gravity_torques:
            self.gravity_torques[key] = frommsg1[temp]
            self.actual_effort[key] = frommsg2[temp]
            temp = temp + 1
            # self.limb.set_joint_torques(self.gravity_torques)

    def get_ik_solution(self, rpy_pose):
        quaternion_pose = conversions.list_to_pose_stamped(rpy_pose, "base")
        node = "ExternalTools/" + "right" + "/PositionKinematicsNode/IKService"
        ik_service = rospy.ServiceProxy(node, SolvePositionIK)
        ik_request = SolvePositionIKRequest()
        hdr = Header(stamp=rospy.Time.now(), frame_id="base")
        ik_request.pose_stamp.append(quaternion_pose)
        try:
            rospy.wait_for_service(node, 15.0)  # 5改成了15
            ik_response = ik_service(ik_request)
        except (rospy.ServiceException, rospy.ROSException), error_message:
            rospy.logerr("Service request failed: %r" % (error_message,))
            sys.exit("ERROR - baxter_ik_move - Failed to append pose")

        if ik_response.isValid[0]:
            print("PASS: Valid joint configuration found")
            # convert response to joint position control dictionary
            limb_joints = dict(zip(ik_response.joints[0].name, ik_response.joints[0].position))
            return limb_joints

        else:
            sys.exit("ERROR - baxter_ik_move - No valid joint configuration found")

    def subscribe_to_gravity_compensation(self):
        topic_str = "robot/limb/right/gravity_compensation_torques"
        rospy.Subscriber(topic_str, SEAJointState, self.gravity_callback)

    '''改变控制模式，还不清楚消息怎么发送
    def publish_to_changeto_positon_mode(self):
        topic_str = "/robot/limb/"+"right"+"/joint_command"
        pub = rospy.Publisher(topic_str,JointCommand,queue_size=10)
        pub.publish(,)
    '''

    def shutdown_close(self):
        #self.limb.move_to_neutral()
        print("shutdown.........")

'''最大的类'''
class from_udp(object):
    def __init__(self, ArmName):
        self.controller = JointControl(ArmName)
        self.limb = baxter_interface.Limb(ArmName)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #UDP通信
        self.s.bind(('10.1.1.20', 6666)) #绑定本机的地址，端口号识别程序


        self.angles = self.limb.joint_angles()
        print self.angles
        self.trans_angles_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.real_trans_angles_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.velocities = self.limb.joint_velocities()
        self.trans_velocities_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.real_trans_velocities_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.trans_z2_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.real_z2_alphas_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.pre_alphas = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.cur_alphas = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.sub = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


        self._pre_time = rospy.get_time()
        self._cur_time = self._pre_time

        self.Rate = rospy.Rate(500)
        '''线程'''
        self.thread_result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.real_thread_result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.thread_stop = False
        self.thread = threading.Thread(target=self.commucate)
        self.start = 1
        self.thread.start()

    def commucate(self):
        while not self.thread_stop:
            self._cur_time = rospy.get_time()
            dt = self._cur_time - self._pre_time

            '''需要传递的角度'''
            self.angles = self.limb.joint_angles()
            temp = 0
            for key in self.angles:
                self.trans_angles_list[temp] = self.angles[key]
                temp = temp + 1

            self.real_trans_angles_list[0] = self.trans_angles_list[0]
            self.real_trans_angles_list[1] = self.trans_angles_list[1]
            self.real_trans_angles_list[2] = self.trans_angles_list[5]
            self.real_trans_angles_list[3] = self.trans_angles_list[6]
            self.real_trans_angles_list[4] = self.trans_angles_list[2]
            self.real_trans_angles_list[5] = self.trans_angles_list[3]
            self.real_trans_angles_list[6] = self.trans_angles_list[4]


            # print self.real_trans_angles_list
            for i in range(0, 7):
                self.real_trans_angles_list[i] = self.real_trans_angles_list[i] * 1000.0 + 32768.0

            '''需要传递的速度'''
            self.velocities = self.limb.joint_velocities()
            temp = 0
            for key in self.velocities:
                self.trans_velocities_list[temp] = self.velocities[key]
                temp = temp + 1

            self.real_trans_velocities_list[0] = self.trans_velocities_list[0]
            self.real_trans_velocities_list[1] = self.trans_velocities_list[1]
            self.real_trans_velocities_list[2] = self.trans_velocities_list[5]
            self.real_trans_velocities_list[3] = self.trans_velocities_list[6]
            self.real_trans_velocities_list[4] = self.trans_velocities_list[2]
            self.real_trans_velocities_list[5] = self.trans_velocities_list[3]
            self.real_trans_velocities_list[6] = self.trans_velocities_list[4]
            # print self.real_trans_velocities_list
            for i in range(0, 7):
                self.real_trans_velocities_list[i] = self.real_trans_velocities_list[i] * 1000.0 + 32768.0


            '''需要传递的z2'''

            self.real_z2_alphas_list[0] = self.trans_z2_list[0]
            self.real_z2_alphas_list[1] = self.trans_z2_list[1]
            self.real_z2_alphas_list[2] = self.trans_z2_list[5]
            self.real_z2_alphas_list[3] = self.trans_z2_list[6]
            self.real_z2_alphas_list[4] = self.trans_z2_list[2]
            self.real_z2_alphas_list[5] = self.trans_z2_list[3]
            self.real_z2_alphas_list[6] = self.trans_z2_list[4]


            for i in range(0, 7):
                self.real_z2_alphas_list[i] = self.real_z2_alphas_list[i] * 1000.0 + 32768.0

            self.msg = struct.pack("H", self.start)
            self.msg += struct.pack("7H", self.real_trans_angles_list[0], self.real_trans_angles_list[1], self.real_trans_angles_list[2], self.real_trans_angles_list[3], self.real_trans_angles_list[4], self.real_trans_angles_list[5], self.real_trans_angles_list[6])
            self.msg += struct.pack("7H", self.real_trans_velocities_list[0], self.real_trans_velocities_list[1], self.real_trans_velocities_list[2], self.real_trans_velocities_list[3], self.real_trans_velocities_list[4], self.real_trans_velocities_list[5], self.real_trans_velocities_list[6])
            self.msg += struct.pack("7H", self.real_z2_alphas_list[0], self.real_z2_alphas_list[1], self.real_z2_alphas_list[2], self.real_z2_alphas_list[3], self.real_z2_alphas_list[4], self.real_z2_alphas_list[5], self.real_z2_alphas_list[6])

            self.s.sendto(self.msg, ('10.1.1.21',8001))
            data,addr = self.s.recvfrom(1024)
            for i in range(0, 7):
                self.thread_result[i] = ((ord(data[2 * i]) * 256 + ord(data[2 * i + 1])) - 32768.0) / 1000.0

            self.real_thread_result[0] = self.thread_result[0]
            self.real_thread_result[1] = self.thread_result[1]
            self.real_thread_result[2] = self.thread_result[4]
            self.real_thread_result[3] = self.thread_result[5]
            self.real_thread_result[4] = self.thread_result[6]
            self.real_thread_result[5] = self.thread_result[2]
            self.real_thread_result[6] = self.thread_result[3]

            # print "self.real_thread_result"
            #print self.real_thread_result


            for i in range(0, 7):
                self.pre_alphas[i] = self.cur_alphas[i]
            self._pre_time = self._cur_time
            self.Rate.sleep()

def main():

    rospy.init_node("PID_controller_test")
    Rate = rospy.Rate(200)
    # 类实例化

    udp = from_udp('left')
    endpoint_pose_init = udp.controller.limb.endpoint_pose()
    endpoint_pose = endpoint_pose_init['position']

    pose_init = udp.controller.limb.joint_angles()
    print pose_init

    joint_goal_init = udp.controller.limb.joint_angles()
    joint_angles_goal_list = [[0.938, 1.491, 2.167, -1.348, 1.393, -0.466, -0.586] ,
[0.939, 1.491, 2.169, -1.349, 1.393, -0.464, -0.586] ,
[0.939, 1.492, 2.174, -1.35, 1.395, -0.458, -0.586] ,
[0.94, 1.493, 2.18, -1.351, 1.397, -0.45, -0.586] ,
[0.941, 1.493, 2.188, -1.353, 1.4, -0.44, -0.587] ,
[0.943, 1.494, 2.196, -1.354, 1.402, -0.429, -0.587] ,
[0.945, 1.494, 2.202, -1.355, 1.404, -0.418, -0.587] ,
[0.947, 1.493, 2.209, -1.357, 1.405, -0.408, -0.587] ,
[0.949, 1.493, 2.215, -1.358, 1.407, -0.397, -0.587] ,
[0.95, 1.492, 2.222, -1.359, 1.408, -0.386, -0.587] ,
[0.951, 1.492, 2.23, -1.36, 1.41, -0.375, -0.586] ,
[0.952, 1.492, 2.24, -1.361, 1.413, -0.362, -0.586] ,
[0.952, 1.493, 2.251, -1.363, 1.416, -0.35, -0.586] ,
[0.952, 1.493, 2.263, -1.365, 1.42, -0.336, -0.586] ,
[0.951, 1.494, 2.275, -1.367, 1.423, -0.323, -0.586] ,
[0.951, 1.494, 2.287, -1.369, 1.427, -0.308, -0.586] ,
[0.952, 1.494, 2.298, -1.371, 1.431, -0.294, -0.586] ,
[0.953, 1.494, 2.309, -1.373, 1.434, -0.279, -0.586] ,
[0.954, 1.493, 2.32, -1.375, 1.437, -0.264, -0.586] ,
[0.955, 1.493, 2.331, -1.377, 1.441, -0.249, -0.586] ,
[0.956, 1.493, 2.341, -1.379, 1.444, -0.234, -0.586] ,
[0.957, 1.493, 2.352, -1.381, 1.448, -0.218, -0.586] ,
[0.958, 1.493, 2.363, -1.383, 1.451, -0.203, -0.586] ,
[0.959, 1.494, 2.374, -1.385, 1.455, -0.187, -0.586] ,
[0.96, 1.494, 2.385, -1.387, 1.458, -0.172, -0.586] ,
[0.961, 1.494, 2.396, -1.389, 1.462, -0.156, -0.586] ,
[0.962, 1.494, 2.407, -1.391, 1.465, -0.14, -0.586] ,
[0.963, 1.494, 2.418, -1.393, 1.469, -0.124, -0.587] ,
[0.964, 1.494, 2.43, -1.395, 1.472, -0.108, -0.587] ,
[0.966, 1.494, 2.441, -1.398, 1.475, -0.092, -0.587] ,
[0.966, 1.493, 2.452, -1.4, 1.479, -0.076, -0.587] ,
[0.967, 1.493, 2.463, -1.402, 1.482, -0.06, -0.587] ,
[0.968, 1.493, 2.474, -1.404, 1.486, -0.044, -0.587] ,
[0.968, 1.493, 2.484, -1.406, 1.489, -0.028, -0.587] ,
[0.968, 1.494, 2.495, -1.409, 1.493, -0.012, -0.587] ,
[0.969, 1.494, 2.506, -1.411, 1.496, 0.005, -0.587] ,
[0.97, 1.494, 2.517, -1.413, 1.499, 0.021, -0.587] ,
[0.971, 1.494, 2.529, -1.415, 1.503, 0.037, -0.587] ,
[0.972, 1.494, 2.54, -1.418, 1.506, 0.053, -0.587] ,
[0.973, 1.495, 2.552, -1.42, 1.509, 0.069, -0.587] ,
[0.974, 1.494, 2.563, -1.422, 1.512, 0.085, -0.588] ,
[0.975, 1.494, 2.574, -1.424, 1.515, 0.1, -0.588] ,
[0.976, 1.494, 2.584, -1.426, 1.518, 0.115, -0.588] ,
[0.977, 1.494, 2.593, -1.427, 1.521, 0.13, -0.588] ,
[0.978, 1.494, 2.603, -1.429, 1.524, 0.143, -0.588] ,
[0.978, 1.495, 2.612, -1.431, 1.527, 0.156, -0.588] ,
[0.979, 1.497, 2.621, -1.433, 1.529, 0.169, -0.588] ,
[0.98, 1.498, 2.629, -1.434, 1.532, 0.18, -0.588] ,
[0.98, 1.5, 2.637, -1.436, 1.534, 0.191, -0.588] ,
[0.981, 1.502, 2.643, -1.438, 1.536, 0.202, -0.588] ,
[0.981, 1.503, 2.649, -1.439, 1.539, 0.211, -0.588] ,
[0.981, 1.503, 2.654, -1.441, 1.541, 0.22, -0.588] ,
[0.981, 1.502, 2.658, -1.442, 1.543, 0.229, -0.588] ,
[0.982, 1.5, 2.661, -1.443, 1.545, 0.237, -0.588] ,
[0.982, 1.498, 2.663, -1.443, 1.546, 0.244, -0.588] ,
[0.982, 1.496, 2.666, -1.444, 1.548, 0.25, -0.588] ,
[0.982, 1.493, 2.669, -1.444, 1.549, 0.256, -0.588] ,
[0.982, 1.491, 2.672, -1.445, 1.55, 0.26, -0.588] ,
[0.982, 1.488, 2.675, -1.445, 1.55, 0.264, -0.588] ,
[0.982, 1.486, 2.677, -1.446, 1.551, 0.268, -0.588] ,
[0.983, 1.485, 2.68, -1.446, 1.552, 0.271, -0.588] ,
[0.983, 1.485, 2.683, -1.447, 1.552, 0.273, -0.587] ,
[0.983, 1.485, 2.685, -1.448, 1.553, 0.276, -0.587] ,
[0.983, 1.486, 2.687, -1.448, 1.554, 0.278, -0.587] ,
[0.983, 1.486, 2.689, -1.449, 1.554, 0.28, -0.587] ,
[0.984, 1.487, 2.69, -1.45, 1.555, 0.282, -0.587] ,
[0.984, 1.487, 2.691, -1.451, 1.556, 0.283, -0.587] ,
[0.984, 1.486, 2.691, -1.451, 1.556, 0.285, -0.587] ,
[0.984, 1.485, 2.691, -1.452, 1.557, 0.286, -0.587] ,
[0.984, 1.485, 2.691, -1.452, 1.557, 0.287, -0.587] ,
[0.984, 1.484, 2.691, -1.453, 1.558, 0.288, -0.588] ,
[0.985, 1.484, 2.692, -1.454, 1.558, 0.289, -0.588] ,
[0.985, 1.485, 2.692, -1.454, 1.559, 0.29, -0.588] ,
[0.985, 1.485, 2.692, -1.455, 1.559, 0.29, -0.588] ,
[0.985, 1.486, 2.693, -1.455, 1.559, 0.291, -0.588] ,
[0.985, 1.486, 2.693, -1.455, 1.56, 0.291, -0.588] ,
[0.985, 1.486, 2.693, -1.455, 1.56, 0.291, -0.587] ,
[0.985, 1.486, 2.693, -1.455, 1.56, 0.291, -0.587] ,
[0.985, 1.485, 2.693, -1.455, 1.56, 0.291, -0.587] ,
[0.985, 1.485, 2.693, -1.455, 1.56, 0.291, -0.587] ,
[0.985, 1.485, 2.693, -1.455, 1.56, 0.291, -0.587] ,
[0.985, 1.485, 2.693, -1.455, 1.561, 0.291, -0.587] ,
[0.985, 1.485, 2.693, -1.456, 1.562, 0.291, -0.587] ,
[0.985, 1.485, 2.693, -1.456, 1.562, 0.291, -0.587] ,
[0.984, 1.486, 2.693, -1.456, 1.563, 0.291, -0.587] ,
[0.984, 1.486, 2.693, -1.456, 1.563, 0.291, -0.588] ,
[0.984, 1.485, 2.693, -1.456, 1.563, 0.291, -0.587] ,
[0.985, 1.485, 2.693, -1.456, 1.562, 0.291, -0.587] ,
[0.985, 1.484, 2.693, -1.455, 1.56, 0.291, -0.587] ,
[0.985, 1.485, 2.693, -1.454, 1.562, 0.291, -0.587] ,
[0.986, 1.485, 2.693, -1.453, 1.562, 0.292, -0.586] ,
[0.986, 1.485, 2.693, -1.451, 1.562, 0.292, -0.587] ,
[0.987, 1.485, 2.693, -1.456, 1.562, 0.292, -0.587] ,
[0.988, 1.485, 2.693, -1.456, 1.562, 0.293, -0.587] ,
[0.989, 1.485, 2.693, -1.456, 1.562, 0.294, -0.587] ,
[0.991, 1.485, 2.693, -1.456, 1.562, 0.295, -0.587] ,
[0.993, 1.485, 2.693, -1.456, 1.562, 0.296, -0.587] ,
[0.995, 1.485, 2.692, -1.456, 1.562, 0.297, -0.587] ,
[0.997, 1.485, 2.692, -1.456, 1.562, 0.299, -0.587] ,
[1.0, 1.485, 2.692, -1.456, 1.562, 0.3, -0.587] ,
[1.003, 1.485, 2.692, -1.456, 1.562, 0.302, -0.587] ]
    joint_vel_goal_list =[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
[0.016, 0.023, 0.172, -0.036, 0.06, 0.206, -0.0] ,
[0.033, 0.036, 0.293, -0.06, 0.1, 0.363, -0.0] ,
[0.05, 0.038, 0.365, -0.074, 0.12, 0.471, -0.01] ,
[0.069, 0.029, 0.388, -0.078, 0.121, 0.53, -0.0] ,
[0.088, 0.009, 0.36, -0.07, 0.1, 0.54, -0.0] ,
[0.101, -0.012, 0.322, -0.059, 0.077, 0.53, 0.0] ,
[0.103, -0.023, 0.311, -0.053, 0.067, 0.53, 0.004] ,
[0.094, -0.025, 0.328, -0.053, 0.07, 0.541, 0.0] ,
[0.073, -0.018, 0.374, -0.057, 0.086, 0.561, 0.0] ,
[0.041, -0.001, 0.447, -0.066, 0.116, 0.591, 0.0] ,
[0.009, 0.016, 0.522, -0.076, 0.148, 0.624, 0.0] ,
[-0.009, 0.026, 0.575, -0.086, 0.17, 0.654, 0.0] ,
[-0.015, 0.027, 0.603, -0.093, 0.184, 0.679, 0.003] ,
[-0.007, 0.021, 0.608, -0.1, 0.188, 0.701, 0.0] ,
[0.013, 0.007, 0.59, -0.105, 0.184, 0.719, -0.0] ,
[0.036, -0.008, 0.564, -0.108, 0.176, 0.734, -0.0] ,
[0.052, -0.016, 0.544, -0.11, 0.171, 0.746, -0.005] ,
[0.061, -0.018, 0.532, -0.109, 0.169, 0.755, -0.005] ,
[0.062, -0.014, 0.527, -0.106, 0.169, 0.762, -0.004] ,
[0.056, -0.003, 0.529, -0.101, 0.173, 0.765, -0.002] ,
[0.048, 0.009, 0.535, -0.097, 0.176, 0.768, 0.0] ,
[0.043, 0.015, 0.541, -0.094, 0.178, 0.771, 0.001] ,
[0.042, 0.017, 0.548, -0.094, 0.178, 0.776, 0.001] ,
[0.045, 0.014, 0.554, -0.095, 0.176, 0.782, 0.0] ,
[0.051, 0.005, 0.562, -0.099, 0.173, 0.789, -0.002] ,
[0.057, -0.004, 0.567, -0.104, 0.169, 0.796, -0.004] ,
[0.059, -0.009, 0.568, -0.107, 0.167, 0.801, -0.0] ,
[0.056, -0.012, 0.566, -0.109, 0.167, 0.804, -0.0] ,
[0.05, -0.011, 0.559, -0.11, 0.168, 0.805, -0.004] ,
[0.039, -0.007, 0.548, -0.11, 0.17, 0.805, -0.002] ,
[0.028, -0.002, 0.539, -0.11, 0.173, 0.804, 0.0] ,
[0.023, 0.003, 0.535, -0.11, 0.174, 0.804, 0.001] ,
[0.022, 0.006, 0.537, -0.111, 0.174, 0.806, 0.001] ,
[0.026, 0.009, 0.545, -0.113, 0.172, 0.808, 0.0] ,
[0.035, 0.012, 0.559, -0.114, 0.169, 0.813, -0.003] ,
[0.045, 0.013, 0.571, -0.116, 0.165, 0.815, -0.006] ,
[0.051, 0.011, 0.576, -0.115, 0.162, 0.813, -0.0] ,
[0.054, 0.008, 0.573, -0.112, 0.159, 0.806, -0.0] ,
[0.055, 0.002, 0.562, -0.106, 0.157, 0.794, -0.0] ,
[0.052, -0.005, 0.543, -0.099, 0.155, 0.777, -0.01] ,
[0.047, -0.01, 0.52, -0.092, 0.152, 0.756, -0.0] ,
[0.043, -0.006, 0.499, -0.087, 0.149, 0.732, -0.009] ,
[0.04, 0.006, 0.48, -0.085, 0.145, 0.703, -0.007] ,
[0.037, 0.027, 0.462, -0.085, 0.139, 0.67, -0.006] ,
[0.035, 0.056, 0.446, -0.087, 0.132, 0.633, -0.004] ,
[0.032, 0.081, 0.427, -0.089, 0.126, 0.596, -0.002] ,
[0.029, 0.091, 0.398, -0.089, 0.12, 0.561, 0.0] ,
[0.026, 0.084, 0.361, -0.085, 0.115, 0.529, 0.001] ,
[0.021, 0.062, 0.315, -0.079, 0.112, 0.5, 0.002] ,
[0.016, 0.023, 0.261, -0.069, 0.109, 0.473, 0.003] ,
[0.011, -0.02, 0.208, -0.059, 0.106, 0.445, 0.004] ,
[0.008, -0.056, 0.169, -0.049, 0.099, 0.414, 0.004] ,
[0.005, -0.086, 0.144, -0.041, 0.09, 0.378, 0.004] ,
[0.004, -0.109, 0.132, -0.034, 0.078, 0.337, 0.004] ,
[0.005, -0.126, 0.134, -0.027, 0.063, 0.293, 0.004] ,
[0.006, -0.133, 0.141, -0.022, 0.048, 0.249, 0.004] ,
[0.007, -0.127, 0.145, -0.02, 0.038, 0.213, 0.004] ,
[0.008, -0.11, 0.145, -0.021, 0.031, 0.182, 0.004] ,
[0.009, -0.08, 0.142, -0.024, 0.029, 0.159, 0.004] ,
[0.011, -0.039, 0.136, -0.03, 0.031, 0.141, 0.005] ,
[0.012, 0.001, 0.126, -0.036, 0.034, 0.128, 0.005] ,
[0.012, 0.026, 0.112, -0.04, 0.036, 0.116, 0.005] ,
[0.012, 0.034, 0.094, -0.041, 0.036, 0.105, 0.004] ,
[0.011, 0.026, 0.073, -0.04, 0.035, 0.095, 0.002] ,
[0.01, 0.003, 0.048, -0.037, 0.032, 0.086, 0.0] ,
[0.009, -0.023, 0.025, -0.033, 0.029, 0.077, -0.003] ,
[0.008, -0.036, 0.009, -0.031, 0.027, 0.069, -0.004] ,
[0.007, -0.039, 0.001, -0.029, 0.025, 0.061, -0.005] ,
[0.006, -0.029, -0.001, -0.028, 0.025, 0.053, -0.005] ,
[0.005, -0.008, 0.005, -0.029, 0.024, 0.045, -0.004] ,
[0.005, 0.014, 0.013, -0.029, 0.024, 0.037, -0.002] ,
[0.004, 0.027, 0.017, -0.027, 0.022, 0.03, -0.001] ,
[0.003, 0.029, 0.019, -0.023, 0.019, 0.023, 0.001] ,
[0.003, 0.023, 0.018, -0.017, 0.014, 0.016, 0.002] ,
[0.002, 0.006, 0.013, -0.009, 0.009, 0.01, 0.004] ,
[0.002, -0.012, 0.008, -0.002, 0.004, 0.005, 0.005] ,
[0.001, -0.021, 0.004, 0.002, 0.003, 0.001, 0.005] ,
[0.0, -0.023, 0.001, 0.003, 0.006, -0.003, 0.004] ,
[-0.002, -0.018, 0.0, 0.001, 0.012, -0.005, 0.002] ,
[-0.003, -0.004, 0.0, -0.003, 0.021, -0.005, -0.001] ,
[-0.005, 0.01, 0.0, -0.008, 0.029, -0.005, -0.004] ,
[-0.005, 0.017, 0.0, -0.011, 0.031, -0.005, -0.005] ,
[-0.004, 0.018, 0.0, -0.01, 0.026, -0.004, -0.005] ,
[-0.003, 0.012, 0.0, -0.006, 0.016, -0.002, -0.003] ,
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
[0.0, -0.0, 0.0, 0.009, -0.0, 0.003, 0.005] ,
[0.0, -0.0, -0.001, 0.0, -0.0, 0.006, 0.0] ,
[0.0, -0.0, -0.001, 0.0, -0.0, 0.01, 0.02] ,
[0.0, -0.0, -0.002, 0.0, -0.0, 0.0, 0.03] ,
[0.0, -0.0, -0.002, 0.0, -0.0, 0.0, 0.0] ,
[0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0] ,
[0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0] ,
[0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0] ,
[0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0] ,
[0.0, -0.0, -0.007, 0.0, -0.0, 0.0, 0.0] ,
[0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0] ,
[0.0, -0.0, -0.01, 0.0, -0.0, 0.0, 0.0] ,
[0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0] ,
[0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0] ,
[0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0] ,
]


    point_sum = len(joint_angles_goal_list)
    point_now = 0


    joint_angles_now = pose_init
    joint_angles_now_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    joint_velocities_now = udp.controller.limb.joint_velocities()
    joint_velocities_now_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    z1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    z2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    alpha = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    dy_tau = udp.controller.limb.joint_efforts()

    cur_time = rospy.get_time()
    pre_time = cur_time

    Time = 10
    count = 1500
    ratio = count / Time
    step_size = Time / count
    output_size = 100
    out_ratio = count / output_size


    '''作图用'''
    '''关节空间'''
    joint_effort_display = np.zeros((7, output_size+1), dtype=float)
    joint_actual_pose_display = np.zeros((7, output_size+1), dtype=float)
    joint_req_pose_display = np.zeros((7, output_size+1), dtype=float)
    tout = np.zeros((7, output_size+1), dtype=float)


    '''输出用计数'''
    a = 0
    cur_time = rospy.get_time()
    pre_time = cur_time

    # temp = 0
    # for key in joint_angles_goal[0]:
    #     joint_angles_goal_list[temp] = joint_angles_goal[0][key]
    #     temp += 1


    for i in range(0, count):
        if not rospy.is_shutdown():
            '''得到角度'''
            joint_angles_now = udp.controller.limb.joint_angles()
            temp = 0
            for key in joint_angles_now:
                joint_angles_now_list[temp] = joint_angles_now[key]
                temp = temp + 1

            '''得到速度'''
            joint_velocities_now = udp.controller.limb.joint_velocities()
            temp = 0
            for key in joint_velocities_now:
                joint_velocities_now_list[temp] = joint_velocities_now[key]
                temp = temp + 1


            '''计算出当前应该的目标点'''
            if point_now < point_sum:
                if i%6 == 0:
                    point_now = point_now + 1
            # if point_now < point_sum:
            #     if abs(joint_angles_goal_list[point_now][0] - joint_angles_now_list[0]) < 0.1 and \
            #         abs(joint_angles_goal_list[point_now][1] - joint_angles_now_list[1]) < 0.1 and\
            #         abs(joint_angles_goal_list[point_now][2] - joint_angles_now_list[2]) < 0.1 and\
            #         abs(joint_angles_goal_list[point_now][6] - joint_angles_now_list[6]) < 0.1:
            #         point_now = point_now + 1

            dy_tau = udp.controller.torquecommand(joint_angles_goal_list[point_now-1])
            '''计算z1'''
           #  for aaa in range(0, 7):
           #      z1[aaa] = joint_angles_now_list[aaa] - joint_angles_goal_list[point_now-1][aaa]
           #
           #  '''计算alpha'''
           #  for aaa in range(0, 7):
           #      alpha[aaa] = -udp.controller.torController[aaa].get_kp() * z1[aaa] + joint_vel_goal_list[point_now-1][aaa]
           #
           #  '''计算z2'''
           #  for aaa in range(0, 7):
           #      z2[aaa] = joint_velocities_now_list[aaa] - alpha[aaa]
           #      udp.trans_z2_list[aaa] = z2[aaa]
           #
           # # print z2
           #  '''得到通信计算的值'''
           #  temp = 0
           #  for key in dy_tau:
           #      dy_tau[key] = -z1[temp] - udp.controller.torController[temp].get_kd() * z2[temp] + udp.real_thread_result[temp]
           #      temp = temp + 1
            #print dy_tau

                # if key == "right_s0" or key == "right_s1" or key == "right_e0" or key == "right_e1":
                #     if dy_tau[key] > 20:
                #         dy_tau[key] = 20
                #     elif dy_tau[key] < -20:
                #         dy_tau[key] = -20
                #     else:
                #         pass
                # else:
                #     if dy_tau[key] > 12:
                #         dy_tau[key] = 12
                #     elif dy_tau[key] < -12:
                #         dy_tau[key] = -12
                #     else:
                #         pass
            if dy_tau['left_s0']>5:
                dy_tau['left_s0'] = 5
            elif dy_tau['left_s0']<-5:
                dy_tau['left_s0'] = -4
            else:
                pass



            if a == 0:
                temp = 0
                start_time = rospy.get_time()
                get_pose = udp.controller.limb.joint_angles()
                for key in get_pose:
                    joint_actual_pose_display[temp, a] = get_pose[key]
                    joint_effort_display[temp, a] = dy_tau[key]
                    #joint_effort_display[temp,a] = tau[(0,temp)]
                    joint_req_pose_display[temp, a] = joint_angles_goal_list[point_now-1][temp]
                    tout[temp, a] = 0
                    temp = temp + 1
                a = a + 1
            print point_now
            # print dy_tau

            udp.controller.limb.set_joint_torques(dy_tau)
            #udp.controller.limb.move_to_joint_positions(joint_angles_goal)

            '''作图用'''
            if i % out_ratio == 0:
                display_cur_time = rospy.get_time()
                '''关节角度 '''
                temp = 0
                get_pose = udp.controller.limb.joint_angles()
                for key in get_pose:
                    joint_actual_pose_display[temp, a] = get_pose[key]
                    joint_effort_display[temp, a] = dy_tau[key]
                    # joint_effort_display[temp,a] = tau[(0,temp)]
                    joint_req_pose_display[temp, a] = joint_angles_goal_list[point_now-1][temp]
                    tout[temp, a] = float(display_cur_time - start_time)
                    temp = temp + 1
                a = a + 1
            Rate.sleep()

        rospy.on_shutdown(udp.controller.shutdown_close)
    udp.thread_stop = True
    udp.controller.limb.exit_control_mode()
    udp.controller.limb.move_to_neutral()





    #tout = np.linspace(0, 10, output_size+1)
    '''关节空间'''
    fig1 = plt.figure(1)
    plt.subplot(1, 1, 1)
    plt.title("joint_w0")
    plt.plot(tout[0].T, joint_actual_pose_display[0],linewidth =3,color="red", label="actual value")
    plt.plot(tout[0].T, joint_req_pose_display[0],linewidth =3,color="green", label="desired value")
    plt.plot(tout[0].T, joint_req_pose_display[0]-joint_actual_pose_display[0],linewidth =3, color="blue", label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.legend(loc='best')

    fig2 = plt.figure(2)
    plt.subplot(1, 1, 1)
    plt.title("joint_w1")
    plt.plot(tout[1].T, joint_actual_pose_display[1],linewidth =3,color="red", label="actual value")
    plt.plot(tout[1].T, joint_req_pose_display[1],linewidth =3,color="green", label="desired value")
    plt.plot(tout[0].T, joint_req_pose_display[1] - joint_actual_pose_display[1],linewidth =3, color="blue", label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.legend(loc='best')


    fig3 = plt.figure(3)
    plt.subplot(1, 1, 1)
    plt.title("joint_w2")
    plt.plot(tout[2].T, joint_actual_pose_display[2],linewidth =3,color="red", label="actual value")
    plt.plot(tout[2].T, joint_req_pose_display[2],linewidth =3,color="green", label="desired value")
    plt.plot(tout[0].T, joint_req_pose_display[2] - joint_actual_pose_display[2],linewidth =3, color="blue", label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.legend(loc='best')

    fig4 = plt.figure(4)
    plt.subplot(1, 1, 1)
    plt.title("joint_e0")
    plt.plot(tout[3].T, joint_actual_pose_display[3],linewidth =3,color="red", label="actual value")
    plt.plot(tout[3].T, joint_req_pose_display[3],linewidth =3, color="green", label="desired value")
    plt.plot(tout[0].T, joint_req_pose_display[3] - joint_actual_pose_display[3],linewidth =3, color="blue", label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.legend(loc='best')


    fig5 = plt.figure(5)
    plt.subplot(1, 1, 1)
    plt.title("joint_e1")
    plt.plot(tout[4].T, joint_actual_pose_display[4],linewidth =3, color="red", label="actual value")
    plt.plot(tout[4].T, joint_req_pose_display[4],linewidth =3,color="green", label="desired value")
    plt.plot(tout[0].T, joint_req_pose_display[4] - joint_actual_pose_display[4],linewidth =3, color="blue", label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.legend(loc='best')


    fig6 = plt.figure(6)
    plt.subplot(1, 1, 1)
    plt.title("joint_s0")
    plt.plot(tout[5].T, joint_actual_pose_display[5],linewidth =3,color="red", label="actual value")
    plt.plot(tout[5].T, joint_req_pose_display[5],linewidth =3,color="green", label="desired value")
    plt.plot(tout[0].T, joint_req_pose_display[5] - joint_actual_pose_display[5],linewidth =3, color="blue", label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.legend(loc='best')


    fig7 = plt.figure(7)
    plt.subplot(1, 1, 1)
    plt.title("joint_s1")
    plt.plot(tout[6].T, joint_actual_pose_display[6],linewidth =3,color="red", label="actual value")
    plt.plot(tout[6].T, joint_req_pose_display[6],linewidth =3,color="green", label="desired value")
    plt.plot(tout[0].T, joint_req_pose_display[6] - joint_actual_pose_display[6],linewidth =3, color="blue", label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.legend(loc='best')
    plt.show()

    fig8 = plt.figure(8)
    plt.subplot(1, 1, 1)
    plt.title("torques")
    plt.plot(tout[0].T, joint_effort_display[0],linewidth =3, label = 'joint_w0')
    plt.plot(tout[0].T, joint_effort_display[1],linewidth =3, label='joint_w1')
    plt.plot(tout[0].T, joint_effort_display[2],linewidth =3, label='joint_w2')
    plt.plot(tout[0].T, joint_effort_display[3],linewidth =3, label='joint_e0')
    plt.plot(tout[0].T, joint_effort_display[4],linewidth =3, label='joint_e1')
    plt.plot(tout[0].T, joint_effort_display[5],linewidth =3, label='joint_s0')
    plt.plot(tout[0].T, joint_effort_display[6],linewidth =3, label='joint_s1')
    plt.xlabel("time/s")
    plt.ylabel("torque/Nm")
    plt.legend(loc='best', bbox_to_anchor=(1, 0.7))
    fig8.savefig('123456.png', transparent=True)
    plt.show()

if __name__ == '__main__':
    main()