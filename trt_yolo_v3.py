#!/usr/bin/env python3

import os
import time

import cv2
import pycuda.autoinit  # For initializing CUDA driver
import pycuda.driver as cuda

from utils.yolo_classes import get_cls_dict
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

import rospy
import rospkg
from yolo_trt_ros.msg import Detector2DArray
from yolo_trt_ros.msg import Detector2D
from yolo_trt_ros.msg import BoundingBox2D
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
from cv_bridge import CvBridge, CvBridgeError


class yolo(object):
    def __init__(self):
        """ Constructor """

        self.bridge = CvBridge()
        self.init_params()
        self.init_yolo()
        self.cuda_ctx = cuda.Device(0).make_context()
        self.trt_yolo = TrtYOLO(
            (self.model_path + self.model), (self.h, self.w), self.category_num)

    def __del__(self):
        """ Destructor """

        self.cuda_ctx.pop()
        del self.trt_yolo
        del self.cuda_ctx

    def clean_up(self):
        """ Backup destructor: Release cuda memory """

        if self.trt_yolo is not None:
            self.cuda_ctx.pop()
            del self.trt_yolo
            del self.cuda_ctx

    def init_params(self):
        """ Initializes ros parameters """
        
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("yolo_trt_ros")
        self.camera_topic_name = rospy.get_param("/yolo_trt_node/subscribers/camera_reading/topic_front", "alphasense_driver_ros/cam4/dropped/debayered")
        self.camera_queue_size = rospy.get_param("/yolo_trt_node/subscribers/camera_reading/queue_size", 1)
        self.img_dim = rospy.get_param("/yolo_trt_node/subscribers/camera_resolution", 720*540*3)
        self.object_detector_topic_name = rospy.get_param("/yolo_trt_node/publishers/object_detector/topic", "/detected_objects")
        self.object_detector_queue_size = rospy.get_param("/yolo_trt_node/publishers/object_detector/queue_size", 1)
        self.bounding_boxes_topic_name = rospy.get_param("/yolo_trt_node/publishers/bounding_boxes/topic", "/bounding_boxes")
        self.bounding_boxes_queue_size = rospy.get_param("/yolo_trt_node/publishers/bounding_boxes/queue_size", 1)
        self.detection_image_topic_name = rospy.get_param("/yolo_trt_node/publishers/detection_image/topic", "/detection_image")
        self.detection_image_queue_size = rospy.get_param("/yolo_trt_node/publishers/detection_image/queue_size", 1)

        self.model = rospy.get_param("/yolo_trt_node/yolo_model/model/name", "yolov3")
        self.model_path = rospy.get_param("yolo_trt_node/yolo_model/model_path", package_path + "/yolo/")
        self.input_shape = rospy.get_param("/yolo_trt_node/yolo_model/input_shape/value", "416")
        self.category_num = rospy.get_param("/yolo_trt_node/yolo_model/category_number/value", 8)
        self.conf_th = rospy.get_param("/yolo_trt_node/yolo_model/confidence_threshold/value", 0.2)
        self.batch_size = rospy.get_param("/yolo_trt_node/yolo_model/batch_size/value", 1)
        self.show_img = rospy.get_param("/yolo_trt_node/image_view/enable_opencv", True)


        self.image_sub = rospy.Subscriber(
            self.camera_topic_name, Image, self.img_callback, queue_size=self.camera_queue_size, buff_size=self.img_dim)
        self.bounding_boxes_publisher = rospy.Publisher(
            self.bounding_boxes_topic_name, Detector2DArray, queue_size=self.bounding_boxes_queue_size)
        self.detection_image_publisher = rospy.Publisher(
            self.detection_image_topic_name, Image, queue_size=self.detection_image_queue_size)
        self.object_publisher = rospy.Publisher(
            self.object_detector_topic_name, Int8, queue_size=self.object_detector_queue_size)
        
        self.iter = 0
        self.avg_fps = 0

    def init_yolo(self):
        """ Initialises yolo parameters required for trt engine """

        if self.model.find('-') == -1:
            self.model = self.model + "-" + self.input_shape

        yolo_dim = self.model.split('-')[-1]

        if 'x' in yolo_dim:
            dim_split = yolo_dim.split('x')
            if len(dim_split) != 2:
                raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
            self.w, self.h = int(dim_split[0]), int(dim_split[1])
        else:
            self.h = self.w = int(yolo_dim)
        if self.h % 32 != 0 or self.w % 32 != 0:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)

        cls_dict = get_cls_dict(self.category_num)
        self.vis = BBoxVisualization(cls_dict)


    def img_callback(self, ros_img):
        """Continuously capture images from camera and do object detection """

        tic = time.time()
        self.iter = self.iter + 1 

        # converts from ros_img to cv_img for processing
        try:
            cv_img = self.bridge.imgmsg_to_cv2(
                ros_img, desired_encoding="bgr8")
            rospy.logdebug("ROS Image converted for processing")
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

        if cv_img is not None:
            boxes, confs, clss = self.trt_yolo.detect(cv_img, self.conf_th, self.batch_size)

            cv_img = self.vis.draw_bboxes(cv_img, boxes, confs, clss)

            toc = time.time()
            fps = 1.0 / (toc - tic)
            self.avg_fps = self.avg_fps*(self.iter-1)/self.iter + fps/self.iter
            print(self.avg_fps)


            self.publisher(boxes, confs, clss)

            if self.show_img:
                cv_img = show_fps(cv_img, fps)
                cv2.imshow("YOLOv3 DETECTION RESULTS", cv_img)
                cv2.waitKey(1)

        # converts back to ros_img type for publishing
        try:
            overlay_img = self.bridge.cv2_to_imgmsg(
                cv_img, encoding="bgr8")
            rospy.logdebug("CV Image converted for publishing")
            self.detection_image_publisher.publish(overlay_img)
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

    def publisher(self, boxes, confs, clss):
        """ Publishes to detector_msgs

        Parameters:
        boxes (List(List(int))) : Bounding boxes of all objects
        confs (List(double))	: Probability scores of all objects
        clss  (List(int))	: Class ID of all classes
        """
        detection2d = Detector2DArray()
        detection = Detector2D()
        detection2d.header.stamp = rospy.Time.now()
        
        self.object_publisher.publish(len(boxes))

        for i in range(len(boxes)):
            # boxes : xmin, ymin, xmax, ymax
            for _ in boxes:
                detection.header.stamp = rospy.Time.now()
                detection.header.frame_id = "camera" # change accordingly
                detection.results.id = clss[i]
                detection.results.score = confs[i]

                detection.bbox.center.x = boxes[i][0] + (boxes[i][2] - boxes[i][0])/2
                detection.bbox.center.y = boxes[i][1] + (boxes[i][3] - boxes[i][1])/2
                detection.bbox.center.theta = 0.0  # change if required

                detection.bbox.size_x = abs(boxes[i][0] - boxes[i][2])
                detection.bbox.size_y = abs(boxes[i][1] - boxes[i][3])

            detection2d.detections.append(detection)
        
        self.bounding_boxes_publisher.publish(detection2d)


def main():
    rospy.init_node('yolo_detection', anonymous=True)
    yolo_ = yolo()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        del yolo_
        rospy.on_shutdown(yolo_.clean_up())
        print("Shutting down")


if __name__ == '__main__':
    main()
