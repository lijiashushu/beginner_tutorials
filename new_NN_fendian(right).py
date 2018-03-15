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

from mpl_toolkits.mplot3d import Axes3D

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
        t2 = [-0.528, -1.049, 1.131, 1.268, 0.618, 1.154, 2.416]
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
        self.torController[0].set_kp(17.7)  # 130#80.0#*0.6
        self.torController[0].set_ki(0.01)
        self.torController[0].set_kd(5.1)  # 10#15#0.01#*0.6#21.0

        self.torController[1].set_kp(15)  # 130#80.0#*0.6
        self.torController[1].set_ki(6)
        self.torController[1].set_kd(18)  # 10#15#0.01#*0.6#21.0

        self.torController[2].set_kp(15.7)  # 130#80.0#*0.6
        self.torController[2].set_ki(0.1)
        self.torController[2].set_kd(1.2)  # 10#15#0.01#*0.6#21.0

        self.torController[3].set_kp(10.02)  # 130#80.0#*0.6
        self.torController[3].set_ki(1.2)
        self.torController[3].set_kd(2.5)  # 10#15#0.01#*0.6#21.0

        self.torController[4].set_kp(10.3)
        self.torController[4].set_ki(0.1) #0.1
        self.torController[4].set_kd(2.1)

        self.torController[5].set_kp(14.6)  # 130#80.0#*0.6
        self.torController[5].set_ki(1.5) #0.05
        self.torController[5].set_kd(2.1)  # 10#15#0.01#*0.6#21.0

        self.torController[6].set_kp(22)  # 130#80.0#*0.6
        self.torController[6].set_ki(1.5)
        self.torController[6].set_kd(4.5)  # 10#15#0.01#*0.6#21.0
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

        self.final_torques['right_w2'] = self.torcmd['right_w2']
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

    udp = from_udp('right')
    endpoint_pose_init = udp.controller.limb.endpoint_pose()
    endpoint_pose = endpoint_pose_init['position']

    pose_init = udp.controller.limb.joint_angles()

    joint_goal_init = udp.controller.limb.joint_angles()
    joint_angles_goal_list = [
[-0.53, -1.053, 1.13, 1.267, 0.618, 1.155, 2.416] ,
[-0.53, -1.053, 1.13, 1.267, 0.618, 1.155, 2.416] ,
[-0.53, -1.053, 1.13, 1.267, 0.618, 1.155, 2.416] ,
[-0.53, -1.052, 1.13, 1.267, 0.618, 1.155, 2.416] ,
[-0.53, -1.052, 1.13, 1.267, 0.619, 1.155, 2.416] ,
[-0.53, -1.052, 1.13, 1.267, 0.619, 1.155, 2.416] ,
[-0.53, -1.051, 1.13, 1.268, 0.619, 1.155, 2.416] ,
[-0.53, -1.051, 1.13, 1.268, 0.619, 1.155, 2.416] ,
[-0.529, -1.05, 1.131, 1.268, 0.619, 1.155, 2.416] ,
[-0.529, -1.05, 1.131, 1.268, 0.619, 1.154, 2.416] ,
[-0.529, -1.05, 1.131, 1.268, 0.618, 1.154, 2.416] ,
[-0.528, -1.049, 1.131, 1.268, 0.618, 1.154, 2.416] ,
[-0.528, -1.049, 1.132, 1.268, 0.617, 1.153, 2.416] ,
[-0.527, -1.049, 1.132, 1.268, 0.617, 1.152, 2.416] ,
[-0.526, -1.048, 1.132, 1.267, 0.616, 1.152, 2.416] ,
[-0.525, -1.048, 1.133, 1.267, 0.615, 1.151, 2.416] ,
[-0.524, -1.048, 1.133, 1.267, 0.614, 1.15, 2.416] ,
[-0.523, -1.047, 1.133, 1.267, 0.614, 1.15, 2.416] ,
[-0.522, -1.047, 1.133, 1.267, 0.614, 1.149, 2.416] ,
[-0.521, -1.047, 1.133, 1.267, 0.614, 1.148, 2.416] ,
[-0.52, -1.046, 1.133, 1.267, 0.614, 1.148, 2.415] ,
[-0.52, -1.046, 1.133, 1.267, 0.614, 1.147, 2.415] ,
[-0.519, -1.045, 1.132, 1.267, 0.616, 1.147, 2.414] ,
[-0.518, -1.045, 1.132, 1.267, 0.617, 1.146, 2.414] ,
[-0.517, -1.044, 1.131, 1.268, 0.619, 1.146, 2.412] ,
[-0.516, -1.043, 1.13, 1.268, 0.622, 1.146, 2.411] ,
[-0.515, -1.041, 1.128, 1.269, 0.626, 1.146, 2.408] ,
[-0.513, -1.039, 1.126, 1.27, 0.63, 1.145, 2.405] ,
[-0.511, -1.037, 1.124, 1.271, 0.634, 1.145, 2.4] ,
[-0.508, -1.034, 1.122, 1.272, 0.64, 1.144, 2.394] ,
[-0.504, -1.03, 1.119, 1.273, 0.646, 1.143, 2.386] ,
[-0.5, -1.025, 1.116, 1.275, 0.653, 1.141, 2.377] ,
[-0.494, -1.02, 1.112, 1.277, 0.661, 1.139, 2.365] ,
[-0.487, -1.013, 1.107, 1.279, 0.669, 1.137, 2.35] ,
[-0.479, -1.005, 1.102, 1.281, 0.679, 1.134, 2.333] ,
[-0.47, -0.996, 1.097, 1.284, 0.689, 1.131, 2.314] ,
[-0.459, -0.986, 1.091, 1.287, 0.7, 1.127, 2.292] ,
[-0.448, -0.975, 1.084, 1.29, 0.712, 1.123, 2.269] ,
[-0.436, -0.964, 1.078, 1.293, 0.724, 1.119, 2.243] ,
[-0.423, -0.952, 1.071, 1.297, 0.736, 1.114, 2.217] ,
[-0.41, -0.939, 1.063, 1.3, 0.749, 1.11, 2.189] ,
[-0.396, -0.926, 1.056, 1.304, 0.762, 1.105, 2.161] ,
[-0.382, -0.913, 1.048, 1.307, 0.776, 1.1, 2.132] ,
[-0.368, -0.9, 1.041, 1.311, 0.789, 1.095, 2.103] ,
[-0.354, -0.887, 1.033, 1.315, 0.802, 1.09, 2.074] ,
[-0.34, -0.875, 1.026, 1.318, 0.815, 1.085, 2.045] ,
[-0.327, -0.862, 1.019, 1.321, 0.827, 1.08, 2.017] ,
[-0.313, -0.85, 1.012, 1.325, 0.84, 1.076, 1.99] ,
[-0.3, -0.838, 1.005, 1.328, 0.852, 1.071, 1.963] ,
[-0.287, -0.826, 0.998, 1.331, 0.864, 1.067, 1.936] ,
[-0.275, -0.814, 0.991, 1.334, 0.876, 1.063, 1.91] ,
[-0.262, -0.803, 0.984, 1.337, 0.888, 1.059, 1.884] ,
[-0.249, -0.791, 0.977, 1.341, 0.9, 1.055, 1.858] ,
[-0.236, -0.78, 0.971, 1.344, 0.912, 1.051, 1.832] ,
[-0.224, -0.768, 0.964, 1.347, 0.923, 1.047, 1.807] ,
[-0.211, -0.757, 0.957, 1.35, 0.935, 1.043, 1.782] ,
[-0.198, -0.746, 0.951, 1.353, 0.947, 1.039, 1.757] ,
[-0.185, -0.735, 0.944, 1.356, 0.958, 1.035, 1.732] ,
[-0.172, -0.723, 0.938, 1.359, 0.97, 1.03, 1.707] ,
[-0.159, -0.712, 0.931, 1.362, 0.982, 1.026, 1.682] ,
[-0.145, -0.701, 0.925, 1.365, 0.993, 1.022, 1.657] ,
[-0.132, -0.69, 0.918, 1.368, 1.005, 1.018, 1.632] ,
[-0.119, -0.679, 0.912, 1.372, 1.017, 1.014, 1.608] ,
[-0.106, -0.668, 0.905, 1.375, 1.028, 1.01, 1.583] ,
[-0.093, -0.657, 0.898, 1.378, 1.04, 1.006, 1.559] ,
[-0.081, -0.646, 0.892, 1.381, 1.052, 1.002, 1.534] ,
[-0.068, -0.635, 0.885, 1.384, 1.064, 0.998, 1.51] ,
[-0.056, -0.625, 0.879, 1.387, 1.076, 0.995, 1.486] ,
[-0.044, -0.614, 0.872, 1.391, 1.088, 0.991, 1.462] ,
[-0.032, -0.604, 0.865, 1.394, 1.101, 0.987, 1.438] ,
[-0.02, -0.593, 0.859, 1.397, 1.113, 0.984, 1.414] ,
[-0.008, -0.583, 0.852, 1.4, 1.125, 0.98, 1.39] ,
[0.004, -0.572, 0.845, 1.403, 1.137, 0.977, 1.366] ,
[0.015, -0.562, 0.839, 1.406, 1.149, 0.973, 1.341] ,
[0.027, -0.551, 0.832, 1.409, 1.161, 0.97, 1.317] ,
[0.039, -0.54, 0.825, 1.412, 1.173, 0.966, 1.292] ,
[0.051, -0.529, 0.819, 1.415, 1.185, 0.962, 1.268] ,
[0.063, -0.518, 0.812, 1.418, 1.197, 0.958, 1.243] ,
[0.075, -0.506, 0.806, 1.421, 1.208, 0.954, 1.217] ,
[0.087, -0.494, 0.799, 1.424, 1.22, 0.95, 1.192] ,
[0.1, -0.483, 0.793, 1.427, 1.231, 0.946, 1.166] ,
[0.112, -0.471, 0.787, 1.43, 1.241, 0.942, 1.141] ,
[0.125, -0.459, 0.781, 1.432, 1.252, 0.938, 1.116] ,
[0.137, -0.448, 0.776, 1.435, 1.261, 0.934, 1.092] ,
[0.148, -0.437, 0.77, 1.437, 1.27, 0.93, 1.069] ,
[0.159, -0.427, 0.765, 1.439, 1.279, 0.926, 1.047] ,
[0.169, -0.417, 0.761, 1.441, 1.286, 0.923, 1.027] ,
[0.179, -0.409, 0.757, 1.443, 1.293, 0.92, 1.008] ,
[0.187, -0.401, 0.754, 1.445, 1.299, 0.917, 0.991] ,
[0.194, -0.395, 0.751, 1.446, 1.304, 0.914, 0.977] ,
[0.2, -0.39, 0.749, 1.447, 1.308, 0.912, 0.965] ,
[0.205, -0.387, 0.747, 1.447, 1.311, 0.911, 0.955] ,
[0.209, -0.384, 0.746, 1.448, 1.313, 0.91, 0.947] ,
[0.212, -0.382, 0.746, 1.448, 1.314, 0.909, 0.941] ,
[0.214, -0.382, 0.745, 1.448, 1.315, 0.908, 0.936] ,
[0.216, -0.381, 0.745, 1.448, 1.315, 0.908, 0.933] ,
[0.217, -0.382, 0.745, 1.448, 1.315, 0.908, 0.93] ,
[0.217, -0.382, 0.746, 1.448, 1.315, 0.908, 0.929] ,
[0.217, -0.383, 0.746, 1.448, 1.315, 0.908, 0.928] ,
[0.217, -0.384, 0.746, 1.448, 1.315, 0.908, 0.928] ,
[0.216, -0.384, 0.747, 1.448, 1.315, 0.909, 0.927]]
    joint_vel_goal_list =[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
[0.0, 0.002, 0.0, 0.001, 0.001, 0.0, 0.0] ,
[0.0, 0.004, 0.001, 0.002, 0.002, 0.0, 0.0] ,
[0.0, 0.006, 0.001, 0.002, 0.003, 0.0, 0.0] ,
[0.001, 0.007, 0.002, 0.003, 0.003, -0.001, 0.0] ,
[0.002, 0.008, 0.003, 0.003, 0.002, -0.001, 0.0] ,
[0.003, 0.009, 0.003, 0.003, 0.001, -0.002, 0.0] ,
[0.004, 0.009, 0.004, 0.002, 0.0, -0.004, 0.0] ,
[0.006, 0.009, 0.004, 0.002, -0.002, -0.005, 0.0] ,
[0.008, 0.009, 0.005, 0.001, -0.005, -0.007, 0.0] ,
[0.01, 0.009, 0.006, 0.0, -0.008, -0.008, 0.001] ,
[0.012, 0.008, 0.007, -0.001, -0.011, -0.01, 0.001] ,
[0.015, 0.008, 0.007, -0.002, -0.014, -0.012, 0.001] ,
[0.017, 0.007, 0.007, -0.003, -0.016, -0.014, 0.001] ,
[0.018, 0.007, 0.007, -0.004, -0.017, -0.015, 0.0] ,
[0.02, 0.007, 0.006, -0.004, -0.016, -0.016, 0.0] ,
[0.021, 0.007, 0.005, -0.004, -0.013, -0.016, -0.001] ,
[0.021, 0.007, 0.003, -0.003, -0.01, -0.016, -0.003] ,
[0.021, 0.008, 0.001, -0.002, -0.004, -0.015, -0.004] ,
[0.021, 0.009, -0.002, -0.001, 0.002, -0.014, -0.006] ,
[0.021, 0.01, -0.005, 0.0, 0.01, -0.013, -0.009] ,
[0.02, 0.011, -0.008, 0.002, 0.019, -0.011, -0.011] ,
[0.018, 0.013, -0.012, 0.004, 0.03, -0.009, -0.014] ,
[0.018, 0.015, -0.017, 0.007, 0.042, -0.007, -0.019] ,
[0.02, 0.02, -0.022, 0.01, 0.055, -0.006, -0.029] ,
[0.025, 0.027, -0.028, 0.013, 0.068, -0.006, -0.043] ,
[0.032, 0.036, -0.035, 0.016, 0.082, -0.008, -0.063] ,
[0.043, 0.047, -0.042, 0.02, 0.097, -0.011, -0.089] ,
[0.056, 0.061, -0.051, 0.024, 0.113, -0.015, -0.119] ,
[0.072, 0.077, -0.06, 0.029, 0.129, -0.021, -0.154] ,
[0.091, 0.095, -0.07, 0.034, 0.146, -0.028, -0.195] ,
[0.113, 0.115, -0.081, 0.039, 0.164, -0.036, -0.24] ,
[0.138, 0.137, -0.092, 0.045, 0.182, -0.045, -0.291] ,
[0.165, 0.162, -0.104, 0.051, 0.201, -0.056, -0.347] ,
[0.194, 0.188, -0.117, 0.057, 0.22, -0.068, -0.406] ,
[0.22, 0.211, -0.128, 0.063, 0.237, -0.078, -0.458] ,
[0.243, 0.231, -0.138, 0.067, 0.252, -0.087, -0.504] ,
[0.262, 0.248, -0.146, 0.071, 0.265, -0.094, -0.544] ,
[0.279, 0.263, -0.153, 0.075, 0.275, -0.1, -0.577] ,
[0.292, 0.274, -0.159, 0.077, 0.283, -0.105, -0.604] ,
[0.302, 0.282, -0.163, 0.079, 0.289, -0.108, -0.624] ,
[0.308, 0.288, -0.166, 0.08, 0.293, -0.11, -0.637] ,
[0.312, 0.29, -0.167, 0.08, 0.294, -0.111, -0.644] ,
[0.312, 0.29, -0.167, 0.079, 0.293, -0.11, -0.645] ,
[0.309, 0.287, -0.165, 0.078, 0.29, -0.108, -0.639] ,
[0.303, 0.281, -0.163, 0.076, 0.284, -0.104, -0.628] ,
[0.297, 0.275, -0.16, 0.074, 0.28, -0.101, -0.617] ,
[0.293, 0.27, -0.157, 0.073, 0.275, -0.098, -0.607] ,
[0.289, 0.266, -0.155, 0.071, 0.271, -0.096, -0.598] ,
[0.286, 0.262, -0.153, 0.07, 0.268, -0.094, -0.589] ,
[0.284, 0.259, -0.151, 0.069, 0.265, -0.092, -0.582] ,
[0.283, 0.256, -0.15, 0.069, 0.263, -0.091, -0.576] ,
[0.282, 0.254, -0.149, 0.068, 0.261, -0.09, -0.57] ,
[0.283, 0.252, -0.148, 0.068, 0.259, -0.09, -0.565] ,
[0.284, 0.251, -0.147, 0.068, 0.259, -0.09, -0.562] ,
[0.286, 0.25, -0.146, 0.068, 0.258, -0.09, -0.559] ,
[0.288, 0.25, -0.146, 0.069, 0.258, -0.09, -0.557] ,
[0.29, 0.25, -0.145, 0.069, 0.258, -0.091, -0.555] ,
[0.292, 0.25, -0.145, 0.069, 0.259, -0.091, -0.554] ,
[0.292, 0.249, -0.145, 0.07, 0.259, -0.091, -0.552] ,
[0.292, 0.248, -0.145, 0.07, 0.26, -0.091, -0.55] ,
[0.292, 0.247, -0.145, 0.07, 0.261, -0.09, -0.548] ,
[0.29, 0.245, -0.145, 0.07, 0.262, -0.09, -0.546] ,
[0.288, 0.244, -0.146, 0.07, 0.263, -0.089, -0.544] ,
[0.285, 0.242, -0.146, 0.07, 0.264, -0.087, -0.542] ,
[0.281, 0.24, -0.146, 0.07, 0.265, -0.086, -0.54] ,
[0.277, 0.238, -0.147, 0.07, 0.267, -0.084, -0.538] ,
[0.272, 0.235, -0.148, 0.07, 0.269, -0.082, -0.536] ,
[0.267, 0.233, -0.148, 0.07, 0.27, -0.081, -0.535] ,
[0.264, 0.232, -0.148, 0.07, 0.271, -0.08, -0.534] ,
[0.261, 0.232, -0.148, 0.069, 0.271, -0.079, -0.535] ,
[0.259, 0.232, -0.148, 0.069, 0.271, -0.079, -0.536] ,
[0.259, 0.234, -0.148, 0.069, 0.27, -0.079, -0.538] ,
[0.259, 0.236, -0.148, 0.068, 0.269, -0.08, -0.54] ,
[0.261, 0.239, -0.147, 0.068, 0.267, -0.081, -0.544] ,
[0.263, 0.243, -0.147, 0.067, 0.264, -0.082, -0.548] ,
[0.266, 0.248, -0.146, 0.067, 0.261, -0.084, -0.553] ,
[0.27, 0.253, -0.145, 0.066, 0.258, -0.087, -0.559] ,
[0.275, 0.26, -0.143, 0.065, 0.254, -0.09, -0.566] ,
[0.279, 0.264, -0.141, 0.064, 0.249, -0.092, -0.569] ,
[0.279, 0.265, -0.138, 0.063, 0.241, -0.093, -0.566] ,
[0.276, 0.262, -0.133, 0.061, 0.232, -0.093, -0.558] ,
[0.271, 0.256, -0.128, 0.058, 0.222, -0.091, -0.544] ,
[0.262, 0.247, -0.121, 0.055, 0.209, -0.089, -0.525] ,
[0.25, 0.235, -0.113, 0.051, 0.195, -0.085, -0.501] ,
[0.236, 0.219, -0.103, 0.047, 0.179, -0.08, -0.47] ,
[0.218, 0.2, -0.093, 0.042, 0.161, -0.074, -0.435] ,
[0.197, 0.177, -0.081, 0.036, 0.142, -0.066, -0.394] ,
[0.174, 0.151, -0.068, 0.03, 0.12, -0.058, -0.347] ,
[0.147, 0.122, -0.054, 0.024, 0.097, -0.048, -0.295] ,
[0.121, 0.094, -0.041, 0.018, 0.075, -0.038, -0.243] ,
[0.097, 0.068, -0.029, 0.012, 0.055, -0.03, -0.197] ,
[0.075, 0.046, -0.019, 0.008, 0.038, -0.022, -0.156] ,
[0.056, 0.028, -0.01, 0.004, 0.024, -0.015, -0.12] ,
[0.04, 0.012, -0.003, 0.001, 0.013, -0.009, -0.088] ,
[0.025, 0.0, 0.002, -0.001, 0.004, -0.005, -0.062] ,
[0.013, -0.009, 0.006, -0.003, -0.002, -0.001, -0.041] ,
[0.004, -0.015, 0.007, -0.003, -0.006, 0.002, -0.025] ,
[-0.003, -0.017, 0.008, -0.003, -0.007, 0.004, -0.014] ,
[-0.008, -0.016, 0.006, -0.002, -0.005, 0.004, -0.008] ,
[-0.011, -0.012, 0.003, 0.0, 0.0, 0.004, -0.006]]


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
    count = 2000
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
    xyz_display = np.zeros((3, output_size + 1), dtype=float)


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
                if i%8 == 0:
                    point_now = point_now + 1
            # if point_now < point_sum:
            #     if abs(joint_angles_goal_list[point_now][0] - joint_angles_now_list[0]) < 0.1 and \
            #         abs(joint_angles_goal_list[point_now][1] - joint_angles_now_list[1]) < 0.1 and\
            #         abs(joint_angles_goal_list[point_now][2] - joint_angles_now_list[2]) < 0.1 and\
            #         abs(joint_angles_goal_list[point_now][6] - joint_angles_now_list[6]) < 0.1:
            #         point_now = point_now + 1


            '''计算z1'''
            for aaa in range(0, 7):
                z1[aaa] = joint_angles_now_list[aaa] - joint_angles_goal_list[point_now-1][aaa]

            '''计算alpha'''
            for aaa in range(0, 7):
                alpha[aaa] = -udp.controller.torController[aaa].get_kp() * z1[aaa] + joint_vel_goal_list[point_now-1][aaa]

            '''计算z2'''
            for aaa in range(0, 7):
                z2[aaa] = joint_velocities_now_list[aaa] - alpha[aaa]
                udp.trans_z2_list[aaa] = z2[aaa]

           # print z2
            '''得到通信计算的值'''
            temp = 0
            for key in dy_tau:
                dy_tau[key] = -z1[temp] - udp.controller.torController[temp].get_kd() * z2[temp] + udp.real_thread_result[temp]
                temp = temp + 1

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
            if a == 0:
                temp = 0
                start_time = rospy.get_time()
                get_pose = udp.controller.limb.joint_angles()
                for key in get_pose:
                    joint_actual_pose_display[temp, a] = get_pose[key]
                    joint_effort_display[temp, a] = dy_tau[key]
                    # joint_effort_display[temp,a] = tau[(0,temp)]
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
                xyz_pose = udp.controller.limb.endpoint_pose()

                xyz_display[0, a] = xyz_pose["position"][0]
                xyz_display[1, a] = xyz_pose["position"][1]
                xyz_display[2, a] = xyz_pose["position"][2]
                for key in get_pose:
                    joint_actual_pose_display[temp, a] = get_pose[key]
                    joint_effort_display[temp, a] = dy_tau[key]
                    # joint_effort_display[temp,a] = tau[(0,temp)]
                    joint_req_pose_display[temp, a] = joint_angles_goal_list[point_now-1][temp]
                    tout[temp, a] = display_cur_time - start_time
                    temp = temp + 1

                a = a + 1
            Rate.sleep()

        rospy.on_shutdown(udp.controller.shutdown_close)
    udp.thread_stop = True
    udp.controller.limb.exit_control_mode()
    #udp.controller.limb.move_to_neutral()
    # udp.controller.limb.move_to_joint_positions(
    #     {'right_s0': 0.14266021327334347, 'right_s1': -0.5292233718204677, 'right_w0': 0.03528155812136451,
    #      'right_w1': 1.4154807720212654, 'right_w2': 0.12808739578843203, 'right_e0': -0.03413107253045045,
    #      'right_e1': 0.7113835903818606})




    #tout = np.linspace(0, 10, output_size+1)
    '''关节空间'''
    fig1 = plt.figure(1)
    plt.subplot(1, 1, 1)
    plt.title("joint_w0")
    plt.plot(tout[0].T, joint_actual_pose_display[0], linewidth=3, color="red", label="actual value")
    plt.plot(tout[0].T, joint_req_pose_display[0], linewidth=3, color="green", label="desired value")
    plt.plot(tout[0].T, joint_req_pose_display[0] - joint_actual_pose_display[0], linewidth=3, color="blue",
             label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.legend(loc='best')

    fig2 = plt.figure(2)
    plt.subplot(1, 1, 1)
    plt.title("joint_w1")
    plt.plot(tout[1].T, joint_actual_pose_display[1], linewidth=3, color="red", label="actual value")
    plt.plot(tout[1].T, joint_req_pose_display[1], linewidth=3, color="green", label="desired value")
    plt.plot(tout[0].T, joint_req_pose_display[1] - joint_actual_pose_display[1], linewidth=3, color="blue",
             label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.legend(loc='best')

    fig3 = plt.figure(3)
    plt.subplot(1, 1, 1)
    plt.title("joint_w2")
    plt.plot(tout[2].T, joint_actual_pose_display[2], linewidth=3, color="red", label="actual value")
    plt.plot(tout[2].T, joint_req_pose_display[2], linewidth=3, color="green", label="desired value")
    plt.plot(tout[0].T, joint_req_pose_display[2] - joint_actual_pose_display[2], linewidth=3, color="blue",
             label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.legend(loc='best')

    fig4 = plt.figure(4)
    plt.subplot(1, 1, 1)
    plt.title("joint_e0")
    plt.plot(tout[3].T, joint_actual_pose_display[3], linewidth=3, color="red", label="actual value")
    plt.plot(tout[3].T, joint_req_pose_display[3], linewidth=3, color="green", label="desired value")
    plt.plot(tout[0].T, joint_req_pose_display[3] - joint_actual_pose_display[3], linewidth=3, color="blue",
             label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.legend(loc='best')

    fig5 = plt.figure(5)
    plt.subplot(1, 1, 1)
    plt.title("joint_e1")
    plt.plot(tout[4].T, joint_actual_pose_display[4], linewidth=3, color="red", label="actual value")
    plt.plot(tout[4].T, joint_req_pose_display[4], linewidth=3, color="green", label="desired value")
    plt.plot(tout[0].T, joint_req_pose_display[4] - joint_actual_pose_display[4], linewidth=3, color="blue",
             label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.legend(loc='best')

    fig6 = plt.figure(6)
    plt.subplot(1, 1, 1)
    plt.title("joint_s0")
    plt.plot(tout[5].T, joint_actual_pose_display[5], linewidth=3, color="red", label="actual value")
    plt.plot(tout[5].T, joint_req_pose_display[5], linewidth=3, color="green", label="desired value")
    plt.plot(tout[0].T, joint_req_pose_display[5] - joint_actual_pose_display[5], linewidth=3, color="blue",
             label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.legend(loc='best')

    fig7 = plt.figure(7)
    plt.subplot(1, 1, 1)
    plt.title("joint_s1")
    plt.plot(tout[6].T, joint_actual_pose_display[6], linewidth=3, color="red", label="actual value")
    plt.plot(tout[6].T, joint_req_pose_display[6], linewidth=3, color="green", label="desired value")
    plt.plot(tout[0].T, joint_req_pose_display[6] - joint_actual_pose_display[6], linewidth=3, color="blue",
             label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.legend(loc='best')


    fig8 = plt.figure(8)
    plt.subplot(1, 1, 1)
    plt.title("torques")
    plt.plot(tout[0].T, joint_effort_display[0], linewidth=3, label='joint_w0')
    plt.plot(tout[0].T, joint_effort_display[1], linewidth=3, label='joint_w1')
    plt.plot(tout[0].T, joint_effort_display[2], linewidth=3, label='joint_w2')
    plt.plot(tout[0].T, joint_effort_display[3], linewidth=3, label='joint_e0')
    plt.plot(tout[0].T, joint_effort_display[4], linewidth=3, label='joint_e1')
    plt.plot(tout[0].T, joint_effort_display[5], linewidth=3, label='joint_s0')
    plt.plot(tout[0].T, joint_effort_display[6], linewidth=3, label='joint_s1')
    plt.xlabel("time/s")
    plt.ylabel("torque/Nm")
    plt.legend(loc='best', bbox_to_anchor=(1, 0.7))
    fig8.savefig('123456.png', transparent=True)


    fig9 = plt.figure(9)
    ax = fig9.add_subplot(111, projection='3d')
    ax.plot(xyz_display[0], xyz_display[1], xyz_display[2], label="actual value",color='red', linewidth=3)
    plt.ylim(-1, 1)
    plt.show()



if __name__ == '__main__':
    main()