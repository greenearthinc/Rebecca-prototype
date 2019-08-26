# rospy for the subscriber
import rospy
import message_filters
# ROS Image message
from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray
#from rospy_tutorials.msg import Floats
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError

from nav_msgs.msg import Odometry
from geometry_msgs.msg import *
from tf.msg import *
# OpenCV2 for saving an image
import cv2
import numpy as np
import time
import scipy.misc
import scipy.signal

# Instantiate CvBridge
bridge = CvBridge()

class imageSourceStereoCam:

	def __init__(self, directory):
		self.curImage = None
		self.curDepth = None
		self.curImageLarge = None
		self.curDepthLarge = None
		self.image_tmp = None
		self.depth_tmp = None
		self.image_cpy = None
		self.depth_cpy = None
		self.resizingRate = None
		#self.file = open('../Odometry.txt', 'a')
		self.file2 = open('../camera_stamps.txt', 'w')
		self.directory = directory
		self.imageLeftCounter = 0
		self.imageRightCounter = 0
		self.imageDepthCounter = 0
		#self.odom_tmp = 0
		
		rospy.init_node('image_listener')

		'''
		image_topic1 = "camera/left_image"
		image_topic2 = "camera/right_image"
		image_topic3 = "camera/depth_map"
		odom_topic4 = "rosaria/pose"
		'''
		image_topic1 = "/camera/color/image_raw"
		#image_topic2 = "/camera/depth/image_rect_raw"
		image_topic3 = "/camera/depth/image_rect_raw"
		#image_topic3 = "/camera/depth/image_rect_raw/compressedDepth"
		#odom_topic4 = "rosaria/pose"
		

		rospy.Subscriber(image_topic1, Image,  callback = self.callback1, queue_size=1)
		#rospy.Subscriber(image_topic2, Image,  callback = self.callback2, queue_size=1)
		rospy.Subscriber(image_topic3, Image,  callback = self.callback3, queue_size=1)
		#rospy.Subscriber(odom_topic4, Odometry, callback = self.callback4, queue_size=1)
		#left_image_sub = message_filters.Subscriber(image_topic1, Image)
		#right_image_sub = message_filters.Subscriber(image_topic2, Image)
		#depth_sub = message_filters.Subscriber(image_topic3, Float32MultiArray)
		#self.ts = message_filters.TimeSynchronizer([left_image_sub, right_image_sub, depth_sub], 10)
		#self.ts.registerCallback(self.callback)
		self.getImage()

	

	def callback1(self, imageLeft):
		self.image_tmp = bridge.imgmsg_to_cv2(imageLeft, "bgr8")
		cv2.imwrite(self.directory+'/left/left'+str(self.imageLeftCounter).zfill(8)+'.jpg', self.image_tmp)

		self.file2 = open('../camera_stamps.txt', 'a')

		self.imageLeftCounter += 1

		self.file2.write("%s %s %s\n"%(imageLeft.header.stamp.secs, imageLeft.header.stamp.nsecs, str(self.imageLeftCounter).zfill(8)))



	'''
	def callback2(self, imageRight):
		imageRight_tmp = bridge.imgmsg_to_cv2(imageRight, "bgr8")
		cv2.imwrite(self.directory+'/right/right'+str(self.imageRightCounter).zfill(8)+'.jpg', imageRight_tmp)
		self.imageRightCounter += 1
	'''
	def callback3(self, depth):
		self.depth_tmp = np.array(bridge.imgmsg_to_cv2(depth, "16UC1")).astype(np.uint16)
		self.depth_tmp[self.depth_tmp>6000] = 6000
		self.depth_tmp[self.depth_tmp<0] = 0
		depth = np.copy(self.depth_tmp)
		depth = (255.0/6000 * depth).astype(np.uint8)
		#depth = (1/1 * depth).astype(np.uint8)
		cv2.imwrite(self.directory+'/depth/depth'+str(self.imageDepthCounter).zfill(8)+'.jpg', depth)
		self.imageDepthCounter += 1

	def getImage(self, img_id = -1):
		while self.image_tmp is None or self.depth_tmp is None:
			pass
		self.curImageLarge = np.copy(self.image_tmp)
		self.curDepthLarge = np.copy(self.depth_tmp)
		imgShape = self.curImageLarge.shape
		self.resizingRate = 150.0/imgShape[1]
		self.curImage = cv2.resize(self.image_tmp,(150, int(self.resizingRate*imgShape[0])), interpolation = cv2.INTER_LINEAR)
		self.image_cpy = np.copy(self.curImageLarge)
		self.curDepth = cv2.resize(self.depth_tmp,(150, int(self.resizingRate*imgShape[0])), interpolation = cv2.INTER_LINEAR)
		self.depth_cpy = np.copy(self.curDepthLarge)
		#print type(self.curImage)
		#cv2.imshow('Image',self.curImage)
		#cv2.imshow('Depth',self.curDepth)
		#cv2.waitKey(30)
		
		
	def reloadImage(self):
		self.curImageLarge = np.copy(self.image_cpy)
		self.curDepthLarge = np.copy(self.depth_cpy)
		
if __name__ == '__main__':
	cam = imageSourceStereoCam("/home/greenearth/Desktop/data")
	for i in range(1000):
		time.sleep(1)
		cam.getImage()
	cam.file2.close()

		
		
