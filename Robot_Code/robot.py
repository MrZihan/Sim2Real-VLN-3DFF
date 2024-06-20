from interbotix_xs_modules.locobot import InterbotixLocobotXS
import socket
import copy
import threading
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np
import rospy
from abc import ABCMeta, abstractmethod
import message_filters
import cv2
import os
from playsound import playsound
import time
import paramiko
import rospy
from std_srvs.srv import *
import pickle
# This script commands arbitrary positions to the pan-tilt servos when using Time-Based-Profile for its Drive Mode
# When operating motors in 'position' control mode, Time-Based-Profile allows you to easily set the duration of a particular movement
#
# To get started, open a terminal and type...
# 'roslaunch interbotix_xslocobot_control xslocobot_python.launch robot_model:=locobot_base'
# Then change to this directory and type 'python pan_tilt_control.py'
    
linear_speed = 0.2
angular_speed = np.pi/3
move_time = 0.1
turn_time = 0.1

height=480
width=640
channel=3

Server_IP = '192.168.101.75'
Robot_IP = '192.168.101.195'

class TCPClient:
    def __init__(self,host:str,port:int):
        self.client=socket.socket()
        start_state = False
        while start_state==False:
            try:
                self.client.connect((host,port))
                start_state = True
                print('Client Start.')
            except:
                time.sleep(1)

    def send_data(self,data:bytes):
        data = pickle.dumps(data)
        self.client.send(pickle.dumps(len(data)).ljust(64))
        self.client.sendall(data)

    def close(self):
        self.client.close()

class TCPServer:
    def __init__(self,host:str,port:int):
        self.server=socket.socket()
        start_state = False
        while start_state==False:
            try:
                self.server.bind((host,port))
                self.server.listen(1)
                start_state = True
                print('Server Start.')
            except:
                time.sleep(1)
        self.client_socket, self.clientAddr = self.server.accept()

    def recv_data(self):
        while True:
            try:
                data_len = self.client_socket.recv(64)
                break
            except:
                time.sleep(1)

        data_len = pickle.loads(data_len)
        buffer = b"" 
        while True:
            received_data = self.client_socket.recv(512)
            buffer = buffer + received_data 
            if len(buffer) == data_len: 
                break
        data = pickle.loads(buffer) 
        return data

    def close(self):
        self.server.close()



class RGBD(object):
    """
    This is a parent class on which the robot
    specific Camera classes would be built.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        """
        Constructor for Camera parent class.
        :param configs: configurations for camera
        :type configs: YACS CfgNode
        """
        self.cv_bridge = CvBridge()
        self.camera_img_lock = threading.RLock()
        self.rgb_img = None
        self.depth_img = None


        rgb_topic = "/locobot/camera/color/image_raw"
        self.rgb_sub = message_filters.Subscriber(rgb_topic, Image)
        depth_topic = "/locobot/camera/aligned_depth_to_color/image_raw"

        self.depth_sub = message_filters.Subscriber(depth_topic, Image)
        img_subs = [self.rgb_sub, self.depth_sub]
        self.sync = message_filters.ApproximateTimeSynchronizer(
            img_subs, queue_size=10, slop=0.2
        )
        self.sync.registerCallback(self._sync_callback)
        while self.rgb_img is None and not rospy.is_shutdown(): 
            pass

    def _sync_callback(self, rgb, depth):
        self.camera_img_lock.acquire()
        try:
            self.rgb_img = self.cv_bridge.imgmsg_to_cv2(rgb, "bgr8")
            self.rgb_img = self.rgb_img[:, :, ::-1]
            self.depth_img = self.cv_bridge.imgmsg_to_cv2(depth, "passthrough")

        except CvBridgeError as e:
            rospy.logerr(e)
        self.camera_img_lock.release()

    def get_rgb(self):
        """
        This function returns the RGB image perceived by the camera.
        :rtype: np.ndarray or None
        """
        self.camera_img_lock.acquire()
        rgb = copy.deepcopy(self.rgb_img)
        self.camera_img_lock.release()
        return rgb

    def get_depth(self):
        """
        This function returns the depth image perceived by the camera.
        
        The depth image is in meters.
        
        :rtype: np.ndarray or None
        """
        self.camera_img_lock.acquire()
        depth = copy.deepcopy(self.depth_img)
        self.camera_img_lock.release()
        if depth is not None:
            depth = depth / 1000.
            ### depth threshold 0.2m - 2m
            depth = depth.reshape(-1)
            #valid = depth > 200
            #valid = np.logical_and(valid, depth < 2000)
            #depth = depth*valid
            depth = depth.reshape(480,640)
            depth_mapped = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.3), cv2.COLORMAP_JET)

        return depth, depth_mapped



def main():

    start_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    locobot = InterbotixLocobotXS(robot_model="locobot_wx250s", use_move_base_action=True)
    rgbd=RGBD()
    rgb_client=TCPClient(Server_IP,5001)
    depth_client=TCPClient(Server_IP,5002)
    location_client=TCPClient(Server_IP,5003)
    action_server = TCPServer(Robot_IP,5000)

    locobot.camera.pan_tilt_go_home() 
    # locobot.base.move_to_pose(0,0,0,wait=True)
    # locobot.base.mb_client.wait_for_result()

    while not rospy.is_shutdown():

        ### robot location [x,y,direction]
        #location = locobot.base.get_locobot_position()
        location = locobot.base.get_odom()
        #### rgbd image
        rgb = rgbd.get_rgb()
        depth, depth_mapped = rgbd.get_depth()       
        
        rgb_client.send_data(rgb)
        depth_client.send_data(depth)
        location_client.send_data(location)
        action = action_server.recv_data()
        print(action)
        #locobot.camera.pan_tilt_go_home() 

        action_type = action[0].item()
        if action_type == -1: # Point Navigation
            locobot.base.move_to_pose(action[1].item(),action[2].item(),action[3].item(),wait=True)
            locobot.base.mb_client.wait_for_result()

        else: # Atomic Actions
            action_value = action[1].item()
            if action_type==0: #forward
                locobot.base.move(linear_speed, 0, move_time*action_value)
                #locobot.base.mb_client.wait_for_result()

            elif action_type==1: #backward
                locobot.base.move(-linear_speed, 0, move_time*action_value)
                #locobot.base.mb_client.wait_for_result()

            elif action_type==2: #left
                locobot.base.move(0, angular_speed, move_time*action_value)
                #locobot.base.mb_client.wait_for_result()

            elif action_type==3 : #right
                locobot.base.move(0, -angular_speed, move_time*action_value)
                #locobot.base.mb_client.wait_for_result()
       
            elif action_type==4:  # camera down 30-degree
                locobot.camera.pan_tilt_move(0, np.pi/6)
                #locobot.base.mb_client.wait_for_result()

            elif action_type==5:  # camera up 30-degree
                locobot.camera.pan_tilt_move(0, -np.pi/6)
                #locobot.base.mb_client.wait_for_result()

            elif action_type==6:  # camera left 30-degree
                locobot.camera.pan_tilt_move(np.pi/6, 0)
                #locobot.base.mb_client.wait_for_result()

            elif action_type==7:  # camera right 30
                locobot.camera.pan_tilt_move(-np.pi/6, 0)
                #locobot.base.mb_client.wait_for_result()

            elif action_type==8:  # camera go home
                locobot.camera.pan_tilt_go_home() 
                #locobot.base.mb_client.wait_for_result()
            else:
                print('Received an incorrect instruction!')

if __name__=='__main__':
    main()
