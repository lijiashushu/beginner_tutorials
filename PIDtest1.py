#!/usr/bin/env python
#coding=utf-8
'''单点目标右臂PID'''
import argparse
import struct
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import baxter_interface
import sys
import rospy
from moveit_commander import conversions

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
from baxter_core_msgs.msg import SEAJointState
from baxter_core_msgs.msg import JointCommand


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

    def compute_output(self, error):
        """
        Performs a PID computation and returns a control value based on
        the elapsed time (dt) and the error signal from a summing junction
        (the error parameter).
        """
        self._cur_time = rospy.get_time()  # get t
        dt = self._cur_time - self._prev_time  # get delta t
        de = error - self._prev_err  # get delta error

        self._cp = error  # proportional term
        self._ci += error * dt  # integral term

        self._cd = 0
        if dt > 0:  # no div by zero
            self._cd = de / dt  # derivative term

        self._prev_time = self._cur_time  # save t for next pass
        self._prev_err = error  # save t-1 error

        return ((self._kp * self._cp) + (self._ki * self._ci) + (self._kd * self._cd))

class JointControl:
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
        #self.limb.set_joint_position_speed(0.1)

        self.actual_effort = self.limb.joint_efforts()
        self.gravity_torques = self.limb.joint_efforts()
        self.final_torques = self.gravity_torques.copy()


        self.qnow = dict()  # 得到关节角度的字典
        self.qnow_value = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])   # 得到角度的值
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
        #最末端
        self.torController[0].set_kp(70)  # 130#80.0#*0.6
        self.torController[0].set_ki(0.01)
        self.torController[0].set_kd(6)  # 10#15#0.01#*0.6#21.0

        self.torController[1].set_kp(120)  # 130#80.0#*0.6
        self.torController[1].set_ki(0.03)
        self.torController[1].set_kd(15)  # 10#15#0.01#*0.6#21.0

        self.torController[2].set_kp(100)  # 130#80.0#*0.6
        self.torController[2].set_ki(0.1)
        self.torController[2].set_kd(4.6)  # 10#15#0.01#*0.6#21.0

        self.torController[3].set_kp(50)  # 130#80.0#*0.6
        self.torController[3].set_ki(0.1)
        self.torController[3].set_kd(3)  # 10#15#0.01#*0.6#21.0

        self.torController[4].set_kp(40)  # 130#80.0#*0.6
        self.torController[4].set_ki(0.1)
        self.torController[4].set_kd(1.8)  # 10#15#0.01#*0.6#21.0

        self.torController[5].set_kp(12)  # 130#80.0#*0.6
        self.torController[5].set_ki(0.05)
        self.torController[5].set_kd(0.6)  # 10#15#0.01#*0.6#21.0

        self.torController[6].set_kp(6)  # 130#80.0#*0.6
        self.torController[6].set_ki(0.05)
        self.torController[6].set_kd(0.8)  # 10#15#0.01#*0.6#21.0

        self.subscribe_to_gravity_compensation()


        '''力矩控制'''

    def torquecommand(self, qd):  # qd为期望轨迹
        self.err = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.qnow = self.limb.joint_angles() #得到每个关节的当前角度

        temp = 0
        for key in self.qnow :
            self.qnow_value[temp] = self.qnow[key]#转化为list
            temp = temp + 1
        #print self.qnow_value
        self.err =qd - self.qnow_value  #计算每个关节角度的误差
        #print self.err
        self.torcmd = self.limb.joint_efforts()
        #print self.torcmd
        temp = 0
        for key in self.torcmd:
            self.torcmd[key] = self.torController[temp].compute_output(self.err[temp])

            temp = temp +1

        self.final_torques = self.gravity_torques
        #self.final_torques['right_w2'] = self.torcmd['right_w2']

        for key in self.torcmd:
            self.final_torques[key] = self.torcmd[key] + self.gravity_torques[key]

        self.limb.set_joint_torques(self.final_torques)


        return self.torcmd

    def gravity_callback(self,data):
        frommsg1 =  data.gravity_model_effort
        frommsg2 =  data.actual_effort

        temp = 0
        for key in self.gravity_torques:
            self.gravity_torques[key] = frommsg1[temp]
            self.actual_effort[key] = frommsg2[temp]
            temp = temp +1

    def get_ik_solution(self,rpy_pose):
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


def main():

    rospy.init_node("PID_controller_test")
    Rate = rospy.Rate(1000)
    #类实例化

    control = JointControl('right')

    endpoint_pose_init = control.limb.endpoint_pose()
    endpoint_pose = endpoint_pose_init['position']
    endpoint_pose_goal = [endpoint_pose[0],endpoint_pose[1]+0.3,endpoint_pose[2]+0.3,-1.0 * math.pi,0.0,0.0]

    pose_init = control.limb.joint_angles()

    joint_angles_goal = pose_init
    joint_angles_goal_list = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])


    Time = 10
    count = 10000
    ratio = count/Time
    step_size = Time/count
    out_ratio = count/100

    #joint_angles_goal = control.get_ik_solution(endpoint_pose_goal)


    #joint_angles_goal['right_w0'] = pose_init['right_w0'] - math.pi
    joint_angles_goal['right_w2'] = pose_init['right_w2'] + math.pi

    '''作图用'''
    joint_effort_display = np.zeros((7,100),dtype=float)
    joint_actual_pose_display = np.zeros((7,100),dtype=float)
    joint_req_pose_display = np.zeros((7,100),dtype=float)

    temp = 0
    for key in joint_angles_goal:
        joint_angles_goal_list[temp] = joint_angles_goal[key]
        temp += 1

    print control.limb.joint_angles()
    print joint_angles_goal_list
    a = 0

    for i in range(1,count):
        print i
        if not rospy.is_shutdown():
            tau = control.torquecommand(joint_angles_goal_list)
            if i%out_ratio == 0:

                get_pose = control.limb.joint_angles()

                temp = 0
                for key in get_pose:
                    test1 = get_pose[key]
                    test2 = joint_actual_pose_display[temp]
                    test3 = joint_actual_pose_display[temp,a]
                    test3 = test1
                    joint_actual_pose_display[temp,a] = get_pose[key]
                    joint_effort_display[temp,a] = tau[key]
                    joint_req_pose_display[temp,a] = joint_angles_goal[key]
                    temp = temp +1

                a = a + 1
            #print tau
            Rate.sleep()

        rospy.on_shutdown(control.shutdown_close)
    control.limb.exit_control_mode()
    #control.limb.move_to_neutral()




    tout = np.linspace(0, 10, 100)
    fig1 = plt.figure(1)
    plt.subplot(4,1,1)
    plt.title("0")
    plt.plot(tout.T, joint_actual_pose_display[0], label='pose_act')
    plt.plot(tout.T, joint_req_pose_display[0], label='pose_req')
    #plt.plot(tout.T, joint_effort_display[0], label='tau')
    plt.legend()

    plt.subplot(4,1,2)
    plt.title("1")
    plt.plot(tout.T, joint_actual_pose_display[1], label='pose_act')
    plt.plot(tout.T, joint_req_pose_display[1], label='pose_req')
    #plt.plot(tout.T, joint_effort_display[1], label='tau')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.title("2")
    plt.plot(tout.T, joint_actual_pose_display[2], label='pose_act')
    plt.plot(tout.T, joint_req_pose_display[2], label='pose_req')
    #plt.plot(tout.T, joint_effort_display[2], label='tau')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.title("3")
    plt.plot(tout.T, joint_actual_pose_display[3], label='pose_act')
    plt.plot(tout.T, joint_req_pose_display[3], label='pose_req')
    #plt.plot(tout.T, joint_effort_display[3], label='tau')
    plt.legend()

    fig2 = plt.figure(2)
    plt.subplot(3, 1, 1)
    plt.title("4")
    plt.plot(tout.T, joint_actual_pose_display[4], label='pose_act')
    plt.plot(tout.T, joint_req_pose_display[4], label='pose_req')
    #plt.plot(tout.T, joint_effort_display[4], label='tau')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.title("5")
    plt.plot(tout.T, joint_actual_pose_display[5], label='pose_act')
    plt.plot(tout.T, joint_req_pose_display[5], label='pose_req')
    #plt.plot(tout.T, joint_effort_display[5], label='tau')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.title("6")
    plt.plot(tout.T, joint_actual_pose_display[6], label='pose_act')
    plt.plot(tout.T, joint_req_pose_display[6], label='pose_req')
    #plt.plot(tout.T, joint_effort_display[6], label='tau')
    plt.legend()

    plt.show()



if __name__ == '__main__':
    main()