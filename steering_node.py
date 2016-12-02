import threading
import numpy as np
import rospy
import cv2
from dbw_mkz_msgs.msg import SteeringCmd
from sensor_msgs.msg import Image, CompressedImage

class SteeringNode(object):
    def __init__(self, get_model_callback, model_callback):
        rospy.init_node('steering_model')
        self.model = get_model_callback()
        self.predict = model_callback
        self.steering = None
        self.img_timestamp = -1
        self.lock = threading.Lock()
        self.image_sub = rospy.Subscriber('/center_camera/image_color', Image,
                                          self.update_image)
        self.image_sub_compressed = rospy.Subscriber('/center_camera/image_color/compressed', CompressedImage,
                                          self.update_image)
        self.pub = rospy.Publisher('/vehicle/steering_cmd',
                                   SteeringCmd, queue_size=1)
        rospy.Timer(rospy.Duration(.02), self.get_steering)

    def update_image(self, img):
        if hasattr(img, 'format') and 'compressed' in img.format:
            buf = np.ndarray(shape=(1, len(img.data)), dtype=np.uint8, buffer=img.data)
            img.data = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
            shape = img.data.shape
        else:
            shape = img.height, img.width, 3
        arr = np.ndarray(shape=shape,
                         dtype=np.uint8,
                         buffer=np.array(img.data))[:,:,::-1]
        timestamp = img.header.stamp.to_nsec()
        with self.lock:
            if self.img_timestamp < timestamp:
                self.img_timestamp = timestamp
                self.steering = self.predict(self.model, arr)

    def get_steering(self, event):
        if self.steering is None:
            return
        message = SteeringCmd()
        message.enable = True
        message.ignore = False
        message.steering_wheel_angle_cmd = self.steering
        self.pub.publish(message)
