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
import moveit_commander
import geometry_msgs.msg
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
        #self.s.bind(('10.1.1.20', 6666)) #绑定本机的地址，端口号识别程序


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

class moveit_trajectory(object):
    def __init__(self, Armname):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander(Armname +"_arm")
        self.limb = baxter_interface.Limb(Armname)

    def plan_target(self, des_position):
        self.group.clear_pose_targets()

        pose_target = geometry_msgs.msg.Pose()
        pose_target.orientation.x = des_position['orientation'][0]
        pose_target.orientation.y = des_position['orientation'][1]
        pose_target.orientation.z = des_position['orientation'][2]
        pose_target.orientation.w = des_position['orientation'][3]
        pose_target.position.x = des_position['position'][0]
        pose_target.position.y = des_position['position'][1]
        pose_target.position.z = des_position['position'][2]
        self.group.set_pose_target(pose_target)
        plan = self.group.plan()

        self.get_trajectory_times = list()
        self.get_trajectory_positions = list()
        self.get_trajectory_velocities = list()
        for i in range(0, len(plan.joint_trajectory.points)):
            self.get_trajectory_positions.append(plan.joint_trajectory.points[i].positions)
            self.get_trajectory_velocities.append(plan.joint_trajectory.points[i].velocities)
            self.get_trajectory_times.append(0.00 + plan.joint_trajectory.points[i].time_from_start.secs + plan.joint_trajectory.points[i].time_from_start.nsecs/1000000000.0)
        # print self.get_trajectory_times

class cubicSpline(object):
    def __init__(self):
        self.outputsize = 101
        self.a = []
        self.b = []
        self.c = []
        self.d = []
        self.f = []
        self.bt = []
        self.gm = []
        self.h = []
        self.x_sample = []
        self.y_sample = []
        self.M = []
        self.sample_count = 0
        self.bound1 = 0
        self.bound2 = 0
        self.result = []

    def initParam(self, count):
        self.a = np.zeros(count, dtype="double")
        self.b = np.zeros(count, dtype="double")
        self.c = np.zeros(count, dtype="double")
        self.d = np.zeros(count, dtype="double")
        self.f = np.zeros(count, dtype="double")
        self.bt = np.zeros(count, dtype="double")
        self.gm = np.zeros(count, dtype="double")
        self.h = np.zeros(count, dtype="double")
        self.M = np.zeros(count, dtype="double")


    def loadData(self, x_data, y_data, count, bound1, bound2):
        if len(x_data) == 0 or len(y_data) == 0 or count < 3:
            return False

        self.initParam(count)

        self.x_sample = x_data
        self.y_sample = y_data
        self.sample_count = count
        self.bound1 = bound1
        self.bound2 = bound2

    def spline(self):

        f1 = self.bound1
        f2 = self.bound2

        for i in range(0, self.sample_count):
            self.b[i] = 2
        for i in range(0, self.sample_count - 1):
            self.h[i] = self.x_sample[i+1] - self.x_sample[i]
        for i in range(0, self.sample_count - 1):
            self.a[i] = self.h[i-1]/(self.h[i-1] + self.h[i])
        self.a[self.sample_count - 1] = 1

        self.c[0] = 1
        for i in range(1, self.sample_count-1):
            self.c[i] = self.h[i] / (self.h[i-1] + self.h[i])



        for i in range(0, self.sample_count-1):
            self.f[i] = (self.y_sample[i+1]-self.y_sample[i])/(self.x_sample[i+1]-self.x_sample[i])

        for i in range(1, self.sample_count-1):
            self.d[i] = 6*(self.f[i]-self.f[i-1])/(self.h[i-1]+self.h[i])

        """追赶法解方程"""
        self.d[0] = 6*(self.f[0] - f1)/self.h[0]
        self.d[self.sample_count - 1] = 6*(f2 - self.f[self.sample_count-2])/self.h[self.sample_count-2]

        self.bt[0] = self.c[0]/self.b[0]
        for i in range(1, self.sample_count-1):
            self.bt[i] = self.c[i]/(self.b[i] - self.a[i]*self.bt[i-1])

        self.gm[0] = self.d[0]/self.b[0]
        for i in range(1, self.sample_count):
            self.gm[i] = (self.d[i] - self.a[i]*self.gm[i-1])/(self.b[i]-self.a[i]*self.bt[i-1])
        self.M[self.sample_count-1] = self.gm[self.sample_count-1]
        temp = self.sample_count - 2
        for i in range(0, self.sample_count-1):
            self.M[temp] = self.gm[temp] - self.bt[temp]*self.M[temp+1]
            temp = temp - 1


    def getYbyX(self, x_in):
        klo = 0
        khi = self.sample_count - 1
        """二分法查找x所在的区间段"""
        while (khi - klo) >1:
            k = (khi+klo)/2
            if self.x_sample[k] > x_in:
                khi = k
            else:
                klo = k
        hh = self.x_sample[khi] - self.x_sample[klo]
        aa = (self.x_sample[khi] - x_in)/hh
        bb = (x_in - self.x_sample[klo])/hh

        y_out = aa * self.y_sample[klo] + bb * self.y_sample[khi] + \
                ((aa*aa*aa-aa)*self.M[klo] + (bb*bb*bb-bb)*self.M[khi])*hh*hh/6.0
        vel = self.M[khi] * (x_in - self.x_sample[klo]) * (x_in - self.x_sample[klo]) / (2 * hh)\
            - self.M[klo] * (self.x_sample[khi] - x_in) * (self.x_sample[khi] - x_in) / (2 * hh)\
            + (self.y_sample[khi] - self.y_sample[klo]) / hh - hh * (self.M[khi] - self.M[klo]) / 6

        return y_out, vel

    '''all_y_data 为得到角度后的转置，所以all_y_data[0]为第一个关节的所有的角度'''
    def caculate(self, all_x_data, all_y_data):
        length = len(all_x_data)
        dis = (all_x_data[length - 1] - all_x_data[0]) / (self.outputsize - 1)
        self.pos_result = np.zeros((self.outputsize, 7), dtype="double")
        self.vel_result = np.zeros((self.outputsize, 7), dtype="double")
        for ii in range(0, 7):
            self.loadData(all_x_data, all_y_data[ii], length, 0, 0)
            self.spline()
            x_out = -dis
            for i in range(0, self.outputsize):
                x_out = x_out + dis
                self.pos_result[i][ii], self.vel_result[i][ii] = self.getYbyX(x_out)

class move_to_position(object):
    def __init__(self, Armname):
        rospy.init_node("PID_controller_test")
        self.Rate = rospy.Rate(200)
        '''插值类'''
        self.insert = cubicSpline()
        '''包括了通信和控制，PD参数在Joint_Control里'''
        self.udp = from_udp(Armname)
        '''moveit轨迹规划，目标为x，y，z 和 旋转四元数'''
        self.moveit = moveit_trajectory(Armname)

        self.draw_picture = True

    def move(self, des_position):

        '''传入目标位置并生成轨迹'''
        self.moveit.plan_target(des_position)

        endpoint_pose_init = self.udp.controller.limb.endpoint_pose()
        endpoint_pose = endpoint_pose_init['position']

        pose_init = self.udp.controller.limb.joint_angles()

        joint_goal_init = self.udp.controller.limb.joint_angles()

        print "it is here!"
        joint_angles_goal_list = list()
        joint_vel_goal_list = list()

        for foo in range(0, len(self.moveit.get_trajectory_positions)):
            joint_angles_goal_list.append(list(self.moveit.get_trajectory_positions[foo]))
            # joint_vel_goal_list.append(list(moveit.get_trajectory_velocities[foo]))

        for foo in range(0, len(joint_angles_goal_list)):
            joint_angles_goal_list[foo][2], joint_angles_goal_list[foo][5] = joint_angles_goal_list[foo][5], \
                                                                             joint_angles_goal_list[foo][2]
            joint_angles_goal_list[foo][3], joint_angles_goal_list[foo][6] = joint_angles_goal_list[foo][6], \
                                                                             joint_angles_goal_list[foo][3]
            joint_angles_goal_list[foo][2], joint_angles_goal_list[foo][3] = joint_angles_goal_list[foo][3], \
                                                                             joint_angles_goal_list[foo][2]
            joint_angles_goal_list[foo][2], joint_angles_goal_list[foo][4] = joint_angles_goal_list[foo][4], \
                                                                             joint_angles_goal_list[foo][2]
            '''没有用moveit得到的速度信息'''
            # joint_vel_goal_list[foo][2], joint_vel_goal_list[foo][5] = joint_vel_goal_list[foo][5], \
            #                                                            joint_vel_goal_list[foo][2]
            # joint_vel_goal_list[foo][3], joint_vel_goal_list[foo][6] = joint_vel_goal_list[foo][6], \
            #                                                            joint_vel_goal_list[foo][3]
            # joint_vel_goal_list[foo][2], joint_vel_goal_list[foo][3] = joint_vel_goal_list[foo][3], \
            #                                                            joint_vel_goal_list[foo][2]
            # joint_vel_goal_list[foo][2], joint_vel_goal_list[foo][4] = joint_vel_goal_list[foo][4], \
            #                                                            joint_vel_goal_list[foo][2]

            '''计算出插值后的所有的点，记过在insert里的pos_result和vel_result'''
            cacul = np.mat(joint_angles_goal_list).T
            data_format_trans = np.array(cacul)
            self.insert.caculate(self.moveit.get_trajectory_times, data_format_trans)

        '''得到点的总数'''
        point_sum = len(self.insert.pos_result)
        point_now = 0

        joint_angles_now = pose_init
        joint_angles_now_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        joint_velocities_now = self.udp.controller.limb.joint_velocities()
        joint_velocities_now_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        z1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        z2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        alpha = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        dy_tau = self.udp.controller.limb.joint_efforts()

        cur_time = rospy.get_time()
        pre_time = cur_time

        count = 8 * (point_sum)
        out_ratio = 4
        output_size = count / out_ratio

        if self.draw_picture == True:
            '''作图用'''
            '''关节空间'''
            joint_effort_display = np.zeros((7, output_size), dtype=float)
            joint_actual_pose_display = np.zeros((7, output_size), dtype=float)
            joint_req_pose_display = np.zeros((7, output_size), dtype=float)
            tout = np.zeros((output_size), dtype=float)
            xyz_display = np.zeros((3, output_size), dtype=float)

        '''循环计数'''
        i = 0
        '''输出用计数'''
        a = 0
        cur_time = rospy.get_time()
        pre_time = cur_time

        while point_now < point_sum:
            if not rospy.is_shutdown():
                '''得到角度'''
                joint_angles_now = self.udp.controller.limb.joint_angles()
                temp = 0
                for key in joint_angles_now:
                    joint_angles_now_list[temp] = joint_angles_now[key]
                    temp = temp + 1

                '''得到速度'''
                joint_velocities_now = self.udp.controller.limb.joint_velocities()
                temp = 0
                for key in joint_velocities_now:
                    joint_velocities_now_list[temp] = joint_velocities_now[key]
                    temp = temp + 1

                '''当前的目标'''
                if i % 8 == 0:
                    point_now = point_now + 1

                '''计算z1'''
                for aaa in range(0, 7):
                    z1[aaa] = joint_angles_now_list[aaa] - self.insert.pos_result[point_now - 1][aaa]

                '''计算alpha'''
                for aaa in range(0, 7):
                    alpha[aaa] = -self.udp.controller.torController[aaa].get_kp() * z1[aaa] + \
                                 self.insert.vel_result[point_now - 1][aaa]

                '''计算z2'''
                for aaa in range(0, 7):
                    z2[aaa] = joint_velocities_now_list[aaa] - alpha[aaa]
                    self.udp.trans_z2_list[aaa] = z2[aaa]

                '''得到通信计算的值'''
                temp = 0
                for key in dy_tau:
                    dy_tau[key] = -z1[temp] - self.udp.controller.torController[temp].get_kd() * z2[temp] + \
                                  self.udp.real_thread_result[temp]
                    temp = temp + 1

                # if i == 0:
                #     start_time = rospy.get_time()
                if self.draw_picture == True:
                    if a == 0:
                        temp = 0
                        start_time = rospy.get_time()
                        tout[a] = 0
                        get_pose = self.udp.controller.limb.joint_angles()
                        xyz_pose = self.udp.controller.limb.endpoint_pose()

                        xyz_display[0, a] = xyz_pose["position"][0]
                        xyz_display[1, a] = xyz_pose["position"][1]
                        xyz_display[2, a] = xyz_pose["position"][2]
                        for key in get_pose:
                            joint_actual_pose_display[temp, a] = get_pose[key]
                            joint_effort_display[temp, a] = dy_tau[key]
                            # joint_effort_display[temp,a] = tau[(0,temp)]
                            joint_req_pose_display[temp, a] = self.insert.pos_result[point_now - 1][temp]

                            temp = temp + 1
                        a = a + 1
                print point_now
                # print dy_tau

                '''给关节力矩'''
                self.udp.controller.limb.set_joint_torques(dy_tau)
                # udp.controller.limb.move_to_joint_positions(joint_angles_goal)

                if self.draw_picture == True:
                    '''作图用'''
                    if i % out_ratio == 0:
                        display_cur_time = rospy.get_time()
                        tout[a] = display_cur_time - start_time
                        '''关节角度 '''
                        temp = 0
                        get_pose = self.udp.controller.limb.joint_angles()
                        xyz_pose = self.udp.controller.limb.endpoint_pose()

                        xyz_display[0, a] = xyz_pose["position"][0]
                        xyz_display[1, a] = xyz_pose["position"][1]
                        xyz_display[2, a] = xyz_pose["position"][2]
                        for key in get_pose:
                            joint_actual_pose_display[temp, a] = get_pose[key]
                            joint_effort_display[temp, a] = dy_tau[key]
                            # joint_effort_display[temp,a] = tau[(0,temp)]
                            joint_req_pose_display[temp, a] = self.insert.pos_result[point_now - 1][temp]
                            temp = temp + 1

                        a = a + 1
                i = i + 1
                self.Rate.sleep()

            rospy.on_shutdown(self.udp.controller.shutdown_close)
        self.udp.thread_stop = True
        self.udp.controller.limb.exit_control_mode()
        # udp.controller.limb.move_to_neutral()
        # udp.controller.limb.move_to_joint_positions(
        #     {'right_s0': 0.14266021327334347, 'right_s1': -0.5292233718204677, 'right_w0': 0.03528155812136451,
        #      'right_w1': 1.4154807720212654, 'right_w2': 0.12808739578843203, 'right_e0': -0.03413107253045045,
        #      'right_e1': 0.7113835903818606})

        if self.draw_picture == True:
            # tout = np.linspace(0, 10, output_size+1)
            '''关节空间'''
            fig1 = plt.figure(1)
            plt.subplot(1, 1, 1)
            plt.title("joint_w0")
            plt.plot(tout.T, joint_actual_pose_display[0], linewidth=3, color="red", label="actual value")
            plt.plot(tout.T, joint_req_pose_display[0], linewidth=3, color="green", label="desired value")
            plt.plot(tout.T, joint_req_pose_display[0] - joint_actual_pose_display[0], linewidth=3, color="blue",
                     label="error value")
            plt.xlabel("time/s")
            plt.ylabel("angle/rad")
            plt.legend(loc='best')

            fig2 = plt.figure(2)
            plt.subplot(1, 1, 1)
            plt.title("joint_w1")
            plt.plot(tout.T, joint_actual_pose_display[1], linewidth=3, color="red", label="actual value")
            plt.plot(tout.T, joint_req_pose_display[1], linewidth=3, color="green", label="desired value")
            plt.plot(tout.T, joint_req_pose_display[1] - joint_actual_pose_display[1], linewidth=3, color="blue",
                     label="error value")
            plt.xlabel("time/s")
            plt.ylabel("angle/rad")
            plt.legend(loc='best')

            fig3 = plt.figure(3)
            plt.subplot(1, 1, 1)
            plt.title("joint_w2")
            plt.plot(tout.T, joint_actual_pose_display[2], linewidth=3, color="red", label="actual value")
            plt.plot(tout.T, joint_req_pose_display[2], linewidth=3, color="green", label="desired value")
            plt.plot(tout.T, joint_req_pose_display[2] - joint_actual_pose_display[2], linewidth=3, color="blue",
                     label="error value")
            plt.xlabel("time/s")
            plt.ylabel("angle/rad")
            plt.legend(loc='best')

            fig4 = plt.figure(4)
            plt.subplot(1, 1, 1)
            plt.title("joint_e0")
            plt.plot(tout.T, joint_actual_pose_display[3], linewidth=3, color="red", label="actual value")
            plt.plot(tout.T, joint_req_pose_display[3], linewidth=3, color="green", label="desired value")
            plt.plot(tout.T, joint_req_pose_display[3] - joint_actual_pose_display[3], linewidth=3, color="blue",
                     label="error value")
            plt.xlabel("time/s")
            plt.ylabel("angle/rad")
            plt.legend(loc='best')

            fig5 = plt.figure(5)
            plt.subplot(1, 1, 1)
            plt.title("joint_e1")
            plt.plot(tout.T, joint_actual_pose_display[4], linewidth=3, color="red", label="actual value")
            plt.plot(tout.T, joint_req_pose_display[4], linewidth=3, color="green", label="desired value")
            plt.plot(tout.T, joint_req_pose_display[4] - joint_actual_pose_display[4], linewidth=3, color="blue",
                     label="error value")
            plt.xlabel("time/s")
            plt.ylabel("angle/rad")
            plt.legend(loc='best')

            fig6 = plt.figure(6)
            plt.subplot(1, 1, 1)
            plt.title("joint_s0")
            plt.plot(tout.T, joint_actual_pose_display[5], linewidth=3, color="red", label="actual value")
            plt.plot(tout.T, joint_req_pose_display[5], linewidth=3, color="green", label="desired value")
            plt.plot(tout.T, joint_req_pose_display[5] - joint_actual_pose_display[5], linewidth=3, color="blue",
                     label="error value")
            plt.xlabel("time/s")
            plt.ylabel("angle/rad")
            plt.legend(loc='best')

            fig7 = plt.figure(7)
            plt.subplot(1, 1, 1)
            plt.title("joint_s1")
            plt.plot(tout.T, joint_actual_pose_display[6], linewidth=3, color="red", label="actual value")
            plt.plot(tout.T, joint_req_pose_display[6], linewidth=3, color="green", label="desired value")
            plt.plot(tout.T, joint_req_pose_display[6] - joint_actual_pose_display[6], linewidth=3, color="blue",
                     label="error value")
            plt.xlabel("time/s")
            plt.ylabel("angle/rad")
            plt.legend(loc='best')

            fig8 = plt.figure(8)
            plt.subplot(1, 1, 1)
            plt.title("torques")
            plt.plot(tout.T, joint_effort_display[0], linewidth=3, label='joint_w0')
            plt.plot(tout.T, joint_effort_display[1], linewidth=3, label='joint_w1')
            plt.plot(tout.T, joint_effort_display[2], linewidth=3, label='joint_w2')
            plt.plot(tout.T, joint_effort_display[3], linewidth=3, label='joint_e0')
            plt.plot(tout.T, joint_effort_display[4], linewidth=3, label='joint_e1')
            plt.plot(tout.T, joint_effort_display[5], linewidth=3, label='joint_s0')
            plt.plot(tout.T, joint_effort_display[6], linewidth=3, label='joint_s1')
            plt.xlabel("time/s")
            plt.ylabel("torque/Nm")
            plt.legend(loc='best', bbox_to_anchor=(1, 0.7))
            fig8.savefig('123456.png', transparent=True)

            fig9 = plt.figure(9)
            ax = fig9.add_subplot(111, projection='3d')
            ax.plot(xyz_display[0], xyz_display[1], xyz_display[2], label="actual value", color='red', linewidth=3)
            plt.ylim(-1, 1)
            plt.show()


def main():

    rospy.init_node("PID_controller_test")
    act = move_to_position("right")
    pose_init = act.udp.limb.endpoint_pose()
    des_pose = dict(position=[1,2,3], orientation = [1,2,3,4])

    print des_pose
    des_pose['position'][0] = pose_init['position'][0]
    des_pose['position'][1] = pose_init['position'][1]+0.2
    des_pose['position'][2] = pose_init['position'][2]+0.2
    des_pose['orientation'][0] = pose_init['orientation'][0]
    des_pose['orientation'][1] = pose_init['orientation'][1]
    des_pose['orientation'][2] = pose_init['orientation'][2]
    des_pose['orientation'][3] = pose_init['orientation'][3]
    act.move(des_pose)

if __name__ == '__main__':
    main()