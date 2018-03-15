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
        self.limb.move_to_neutral()
        # self.limb.set_joint_position_speed(0.1)

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
        self.torController[0].set_kp(6.7)  # 130#80.0#*0.6
        self.torController[0].set_ki(0.01)
        self.torController[0].set_kd(7.1)  # 10#15#0.01#*0.6#21.0

        self.torController[1].set_kp(15)  # 130#80.0#*0.6
        self.torController[1].set_ki(6)
        self.torController[1].set_kd(22)  # 10#15#0.01#*0.6#21.0

        self.torController[2].set_kp(6.7)  # 130#80.0#*0.6
        self.torController[2].set_ki(0.1)
        self.torController[2].set_kd(1)  # 10#15#0.01#*0.6#21.0

        self.torController[3].set_kp(16.02)  # 130#80.0#*0.6
        self.torController[3].set_ki(1.2)
        self.torController[3].set_kd(2.5)  # 10#15#0.01#*0.6#21.0

        self.torController[4].set_kp(10.3)
        self.torController[4].set_ki(0.1) #0.1
        self.torController[4].set_kd(7.1)

        self.torController[5].set_kp(14.6)  # 130#80.0#*0.6
        self.torController[5].set_ki(1.5) #0.05
        self.torController[5].set_kd(4.1)  # 10#15#0.01#*0.6#21.0

        self.torController[6].set_kp(22)  # 130#80.0#*0.6
        self.torController[6].set_ki(1.5)
        self.torController[6].set_kd(6.5)  # 10#15#0.01#*0.6#21.0
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
        self.limb.move_to_neutral()
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

        self.trans_alphas_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.real_trans_alphas_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.trans_d_alphas_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.real_trans_d_alphas_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.z1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.goal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        '''目标为当前角度'''
        lll = self.controller.limb.joint_angles()
        temp = 0
        for key in lll:
            self.goal[temp] = lll[key]
            temp = temp + 1

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


            '''需要传递的alpha'''
            temp = 0
            for key in self.angles:
                self.trans_alphas_list[temp] = -(self.angles[key] - self.goal[temp]) * self.controller.torController[temp].get_kp()
                self.cur_alphas[temp] = self.trans_alphas_list[temp]
                temp = temp +1

            #print self.cur_alphas
            self.real_trans_alphas_list[0] = self.trans_alphas_list[0]
            self.real_trans_alphas_list[1] = self.trans_alphas_list[1]
            self.real_trans_alphas_list[2] = self.trans_alphas_list[5]
            self.real_trans_alphas_list[3] = self.trans_alphas_list[6]
            self.real_trans_alphas_list[4] = self.trans_alphas_list[2]
            self.real_trans_alphas_list[5] = self.trans_alphas_list[3]
            self.real_trans_alphas_list[6] = self.trans_alphas_list[4]

            # print self.real_trans_alphas_list
            for i in range(0, 7):
                self.real_trans_alphas_list[i] = self.real_trans_alphas_list[i] * 100.0 + 32768.0

            '''计算d_alpha'''
            for i in range(0, 7):
                self.sub[i] = self.cur_alphas[i] - self.pre_alphas[i]
                if dt > 0.002:
                    self.trans_d_alphas_list[i] = self.sub[i]/dt
                    '''d_alpha不能大于10'''
                    if self.trans_d_alphas_list[i] > 5:
                        self.trans_d_alphas_list[i] = 5
                    elif self.trans_d_alphas_list[i] < -5:
                        self.trans_d_alphas_list[i] = -5
                    else:
                        pass
                else:
                    self.trans_d_alphas_list[i] = 0

            self.real_trans_d_alphas_list[0] = self.trans_d_alphas_list[0]
            self.real_trans_d_alphas_list[1] = self.trans_d_alphas_list[1]
            self.real_trans_d_alphas_list[2] = self.trans_d_alphas_list[5]
            self.real_trans_d_alphas_list[3] = self.trans_d_alphas_list[6]
            self.real_trans_d_alphas_list[4] = self.trans_d_alphas_list[2]
            self.real_trans_d_alphas_list[5] = self.trans_d_alphas_list[3]
            self.real_trans_d_alphas_list[6] = self.trans_d_alphas_list[4]

            # print self.real_trans_d_alphas_list
            # print "\n\n\n"
            for i in range(0, 7):
                self.real_trans_d_alphas_list[i] = self.real_trans_d_alphas_list[i] * 1000.0 + 32768.0


            self.msg = struct.pack("7H", self.real_trans_angles_list[0], self.real_trans_angles_list[1], self.real_trans_angles_list[2], self.real_trans_angles_list[3], self.real_trans_angles_list[4], self.real_trans_angles_list[5], self.real_trans_angles_list[6])
            self.msg += struct.pack("7H", self.real_trans_velocities_list[0], self.real_trans_velocities_list[1], self.real_trans_velocities_list[2], self.real_trans_velocities_list[3], self.real_trans_velocities_list[4], self.real_trans_velocities_list[5], self.real_trans_velocities_list[6])
            self.msg += struct.pack("7H", self.real_trans_alphas_list[0], self.real_trans_alphas_list[1], self.real_trans_alphas_list[2], self.real_trans_alphas_list[3], self.real_trans_alphas_list[4], self.real_trans_alphas_list[5], self.real_trans_alphas_list[6])
            self.msg += struct.pack("7H", self.real_trans_d_alphas_list[0], self.real_trans_d_alphas_list[1], self.real_trans_d_alphas_list[2], self.real_trans_d_alphas_list[3], self.real_trans_d_alphas_list[4], self.real_trans_d_alphas_list[5], self.real_trans_d_alphas_list[6])

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
            # print self.real_thread_result


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

    joint_angles_goal = pose_init
    joint_angles_goal_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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
    count = 3000
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
    '''直角空间'''
    endpoint_pose_actual_display = np.zeros((7, output_size+1))
    endpoint_req_pose_display = np.zeros((7, output_size+1))

    '''添加路径点'''
    waypoint_count = 0
    waypoint_sum = 1
    endpoint_pose_goal = list()
    endpoint_pose_goal.append([endpoint_pose[0], endpoint_pose[1], endpoint_pose[2], -1.0 * math.pi, 0.0, 0.0])

    for i in range(1, waypoint_sum + 1):
        x_improvment = i * (-0.05) / waypoint_sum
        y_improvment = i * 0.3 / waypoint_sum
        z_improvment = i * (-0.0) / waypoint_sum
        endpoint_pose_goal.append(
            [endpoint_pose[0], endpoint_pose[1] + y_improvment, endpoint_pose[2], -1.0 * math.pi, 0.01, 0.01])

    cofficient = 0.3 / output_size

    '''输出用计数'''
    a = 0
    cur_time = rospy.get_time()
    pre_time = cur_time
    '''分频'''
    div = 4

    for i in range(0, count):
        if not rospy.is_shutdown():

            '''计算出当前应该的目标点'''
            endpoint_pose_now = udp.controller.limb.endpoint_pose()
            if waypoint_count < waypoint_sum:
                if endpoint_pose_goal[waypoint_count][1] - endpoint_pose_now['position'][1] < 0.05:
                    waypoint_count = waypoint_count + 1
                    # print control.limb.endpoint_pose()
                    # print endpoint_pose_goal[waypoint_count]
                    joint_angles_goal = udp.controller.get_ik_solution(endpoint_pose_goal[waypoint_count])
                    temp = 0
                    for key in joint_angles_goal:
                        joint_angles_goal_list[temp] = joint_angles_goal[key]
                        udp.goal[temp] = joint_angles_goal_list[temp]
                        temp += 1



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

            '''计算z1'''
            for aaa in range(0, 7):
                z1[aaa] = joint_angles_now_list[aaa] - joint_angles_goal_list[aaa]

            '''计算alpha'''
            for aaa in range(0, 7):
                alpha[aaa] = -udp.controller.torController[aaa].get_kp() * z1[aaa]

            '''计算z2'''
            for aaa in range(0, 7):
                z2[aaa] = joint_velocities_now_list[aaa] - alpha[aaa]

           # print z2
            '''得到通信计算的值'''
            temp = 0
            for key in dy_tau:
                dy_tau[key] = -z1[temp] - udp.controller.torController[temp].get_kd() * z2[temp] + udp.real_thread_result[temp]
                temp = temp + 1
                if key == "right_s0" or key == "right_s1" or key == "right_e0" or key == "right_e1":
                    if dy_tau[key] > 5:
                        dy_tau[key] = 5
                    elif dy_tau[key] < -5:
                        dy_tau[key] = -5
                    else:
                        pass
                else:
                    if dy_tau[key] > 2:
                        dy_tau[key] = 2
                    elif dy_tau[key] < -2:
                        dy_tau[key] = -2
                    else:
                        pass
            if a == 0:
                temp = 0
                start_time = rospy.get_time()
                get_pose = udp.controller.limb.joint_angles()
                for key in get_pose:
                    joint_actual_pose_display[temp, a] = get_pose[key]
                    joint_effort_display[temp, a] = dy_tau[key]
                    # joint_effort_display[temp,a] = tau[(0,temp)]
                    joint_req_pose_display[temp, a] = joint_angles_goal[key]
                    tout[temp, a] = 0
                    temp = temp + 1
                a = a + 1

            #print dy_tau

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
                    joint_req_pose_display[temp, a] = joint_angles_goal[key]
                    tout[temp, a] = display_cur_time - start_time
                    temp = temp + 1
                a = a + 1
            Rate.sleep()

        rospy.on_shutdown(udp.controller.shutdown_close)
    print a
    udp.thread_stop = True
    udp.controller.limb.exit_control_mode()
    udp.controller.limb.move_to_neutral()




    #tout = np.linspace(0, 10, output_size+1)
    '''关节空间'''
    fig1 = plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.title("0")
    # plt.plot(tout.T, joint_actual_pose_display[0], label='pose_act')
    # plt.plot(tout.T, joint_req_pose_display[0], label='pose_req')
    plt.plot(tout.T,joint_req_pose_display[0]-joint_actual_pose_display[0])
    #plt.ylim((-0.2,0.2))
    # plt.plot(tout.T, joint_effort_display[0], label='tau')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.title("1")
    plt.plot(tout.T, joint_req_pose_display[1] - joint_actual_pose_display[1])
    #plt.plot(tout.T, joint_effort_display[1])
    #plt.ylim((-0.2, 0.2))
    # plt.plot(tout.T, joint_actual_pose_display[1], label='pose_act')
    # plt.plot(tout.T, joint_req_pose_display[1], label='pose_req')
    # plt.plot(tout.T, joint_effort_display[1], label='tau')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.title("2")
    plt.plot(tout.T, joint_req_pose_display[2] - joint_actual_pose_display[2])
    #plt.ylim((-0.2, 0.2))
    # plt.plot(tout.T, joint_actual_pose_display[2], label='pose_act')
    # plt.plot(tout.T, joint_req_pose_display[2], label='pose_req')
    # plt.plot(tout.T, joint_effort_display[2], label='tau')
    plt.legend()

    fig2 = plt.figure(2)
    plt.subplot(3, 1, 1)
    plt.title("3")
    plt.plot(tout.T, joint_req_pose_display[3] - joint_actual_pose_display[3])
   # plt.ylim((-0.2, 0.2))
    # plt.plot(tout.T, joint_actual_pose_display[3], label='pose_act')
    # plt.plot(tout.T, joint_req_pose_display[3], label='pose_req')
    # plt.plot(tout.T, joint_effort_display[3], label='tau')
    plt.legend()

    fig2 = plt.figure(2)
    plt.subplot(3, 1, 2)
    plt.title("4")
    plt.plot(tout.T, joint_req_pose_display[4] - joint_actual_pose_display[4])
   # plt.ylim((-0.2, 0.2))
    # plt.plot(tout.T, joint_actual_pose_display[4], label='pose_act')
    # plt.plot(tout.T, joint_req_pose_display[4], label='pose_req')
    # plt.plot(tout.T, joint_effort_display[4], label='tau')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.title("5")
    plt.plot(tout.T, joint_req_pose_display[5] - joint_actual_pose_display[5])
    #plt.ylim((-0.2, 0.2))
    # plt.plot(tout.T, joint_actual_pose_display[5], label='pose_act')
    # plt.plot(tout.T, joint_req_pose_display[5], label='pose_req')
    # plt.plot(tout.T, joint_effort_display[5], label='tau')
    plt.legend()

    fig3 = plt.figure(3)
    plt.subplot(3, 1, 1)
    plt.title("6")
    plt.plot(tout.T, joint_req_pose_display[6] - joint_actual_pose_display[6])
    #plt.ylim((-0.2, 0.2))
    # plt.plot(tout.T, joint_actual_pose_display[6], label='pose_act')
    # plt.plot(tout.T, joint_req_pose_display[6], label='pose_req')
    # plt.plot(tout.T, joint_effort_display[6], label='tau')
    #plt.legend()

    plt.show()

if __name__ == '__main__':
    main()