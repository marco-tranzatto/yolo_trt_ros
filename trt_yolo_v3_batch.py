#!/usr/bin/env python3

import os
import time

import cv2
import pycuda.autoinit  # For initializing CUDA driver
import pycuda.driver as cuda

from utils.yolo_classes import get_cls_dict
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins_batch import TrtYOLO

import rospy
import rospkg
from yolo_trt_ros.msg import Detector2DArray
from yolo_trt_ros.msg import Detector2D
from vision_msgs.msg import BoundingBox2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int8, Int32


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
        self.video_topic1 = rospy.get_param("/yolo_trt_node_batch/subscribers/camera_reading/topic_front", "alphasense_driver_ros/cam4/dropped/debayered")
        self.video_topic2 = rospy.get_param("/yolo_trt_node_batch/subscribers/camera_reading/topic_left", "alphasense_driver_ros/cam3/dropped/debayered")
        self.video_topic3 = rospy.get_param("/yolo_trt_node_batch/subscribers/camera_reading/topic_right", "alphasense_driver_ros/cam5/dropped/debayered")
        #self.video_topic4 = rospy.get_param("/yolo_trt_node_batch/subscribers/camera_reading/topic_top", "alphasense_driver_ros/cam6/dropped/debayered")
        self.camera_queue_size = rospy.get_param("/yolo_trt_node_batch/subscribers/camera_reading/queue_size", 1)
        self.img_dim = rospy.get_param("/yolo_trt_node_batch/subscribers/camera_resolution", 720*540*3)
        self.object_detector_topic_name = rospy.get_param("/yolo_trt_node_batch/publishers/object_detector/topic", "/detected_objects")
        self.object_detector_queue_size = rospy.get_param("/yolo_trt_node_batch/publishers/object_detector/queue_size", 1)
        self.bounding_boxes_topic_name = rospy.get_param("/yolo_trt_node_batch/publishers/bounding_boxes/topic", "/bounding_boxes")
        self.bounding_boxes_queue_size = rospy.get_param("/yolo_trt_node_batch/publishers/bounding_boxes/queue_size", 1)
        self.detection_image_topic_name1 = rospy.get_param("/yolo_trt_node_batch/publishers/detection_image/topic" + "/front", "/detection_image/front")
        self.detection_image_topic_name2 = rospy.get_param("/yolo_trt_node_batch/publishers/detection_image/topic" + "/left", "/detection_image/left")
        self.detection_image_topic_name3 = rospy.get_param("/yolo_trt_node_batch/publishers/detection_image/topic" + "/right", "/detection_image/right")
        #self.detection_image_topic_name4 = rospy.get_param("/yolo_trt_node_batch/publishers/detection_image/topic" + "/top", "/detection_image4/top")
        self.detection_image_queue_size = rospy.get_param("/yolo_trt_node_batch/publishers/detection_image/queue_size", 1)

        self.model = rospy.get_param("/yolo_trt_node_batch/yolo_model/model/name", "yolov3")
        self.model_path = rospy.get_param("yolo_trt_node_batch/yolo_model/model_path", package_path + "/yolo/")
        self.input_shape = rospy.get_param("/yolo_trt_node_batch/yolo_model/input_shape/value", "416")
        self.category_num = rospy.get_param("/yolo_trt_node_batch/yolo_model/category_number/value", 8)
        self.conf_th = rospy.get_param("/yolo_trt_node_batch/yolo_model/confidence_threshold/value", 0.2)
        self.batch_size = rospy.get_param("/yolo_trt_node_batch/yolo_model/batch_size/value", 1)
        self.show_img = rospy.get_param("/yolo_trt_node_batch/image_view/enable_opencv", True)


        self.image_sub1 = rospy.Subscriber(
            self.video_topic1, Image, self.img_callback1, queue_size=self.camera_queue_size, buff_size=self.img_dim)
        self.image_sub2 = rospy.Subscriber(
            self.video_topic2, Image, self.img_callback2, queue_size=self.camera_queue_size, buff_size=self.img_dim)
        self.image_sub3 = rospy.Subscriber(
            self.video_topic3, Image, self.img_callback3, queue_size=self.camera_queue_size, buff_size=self.img_dim)
        #self.image_sub4 = rospy.Subscriber(
        #    self.video_topic4, Image, self.img_callback4, queue_size=self.camera_queue_size, buff_size=self.img_dim)            
        self.bounding_boxes_publisher = rospy.Publisher(
            self.bounding_boxes_topic_name, Detector2DArray, queue_size=self.bounding_boxes_queue_size)
        self.detection_image_publisher1 = rospy.Publisher(
            self.detection_image_topic_name1, Image, queue_size=self.detection_image_queue_size)
        self.detection_image_publisher2 = rospy.Publisher(
            self.detection_image_topic_name2, Image, queue_size=self.detection_image_queue_size)
        self.detection_image_publisher3 = rospy.Publisher(
            self.detection_image_topic_name3, Image, queue_size=self.detection_image_queue_size)
        #self.detection_image_publisher4 = rospy.Publisher(
        #    self.detection_image_topic_name4, Image, queue_size=self.detection_image_queue_size)
        self.object_publisher = rospy.Publisher(
            self.object_detector_topic_name, Int8, queue_size=self.object_detector_queue_size)
        self.framerate_publisher = rospy.Publisher("/yolo_trt_node/framerate/value", Int32, queue_size=1)
        
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

    def img_callback1(self, ros_img):
        """Continuously capture images from camera and do object detection """

        # converts from ros_img to cv_img for processing
        try:
            cv_img = self.bridge.imgmsg_to_cv2(
                ros_img, desired_encoding="bgr8")
            rospy.logdebug("ROS Image converted for processing")
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

        tic = time.time()
        self.iter = self.iter + 1 

        boxes, confs, clss = self.trt_yolo.detect1(cv_img, self.conf_th)
        cv_img = self.vis.draw_bboxes(cv_img, boxes, confs, clss)
        toc = time.time()
        fps = 1.0 / (toc - tic)
        #print("Cam1: " + str(fps) + "\n")
        self.avg_fps = self.avg_fps*(self.iter-1)/self.iter + fps/self.iter
        print(self.avg_fps)
        #self.framerate_publisher.publish(self.avg_fps)
        self.publisher(boxes, confs, clss, "Front Camera")
        
        
        if self.show_img:
            cv_img = show_fps(cv_img, fps)
            cv2.imshow("Cam 1", cv_img)
            cv2.waitKey(1)
        
        
        # converts back to ros_img type for publishing
        try:
            overlay_img = self.bridge.cv2_to_imgmsg(
                cv_img, encoding="bgr8")
            rospy.logdebug("CV Image converted for publishing")
            self.detection_image_publisher1.publish(overlay_img)
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

    def img_callback2(self, ros_img):
        """Continuously capture images from camera and do object detection """

        # converts from ros_img to cv_img for processing
        # time both cv_bridge fns
        try:
            cv_img = self.bridge.imgmsg_to_cv2(
                ros_img, desired_encoding="bgr8")
            rospy.logdebug("ROS Image converted for processing")
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

        tic = time.time()
        boxes, confs, clss = self.trt_yolo.detect2(cv_img, self.conf_th)
        cv_img = self.vis.draw_bboxes(cv_img, boxes, confs, clss)
        toc = time.time()
        fps = 1.0 / (toc - tic)
        #print("Cam2: " + str(fps) + "\n")
        self.publisher(boxes, confs, clss, "Left Camera")

        # time 1
        """
        if self.show_img:
            cv_img = show_fps(cv_img, fps)
            cv2.imshow("Cam 2", cv_img)
            # time 2
            cv2.waitKey(1)
        """
        
        # converts back to ros_img type for publishing
        try:
            overlay_img = self.bridge.cv2_to_imgmsg(
                cv_img, encoding="bgr8")
            rospy.logdebug("CV Image converted for publishing")
            self.detection_image_publisher2.publish(overlay_img)
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

    def img_callback3(self, ros_img):
        """Continuously capture images from camera and do object detection """

        # converts from ros_img to cv_img for processing
        try:
            cv_img = self.bridge.imgmsg_to_cv2(
                ros_img, desired_encoding="bgr8")
            rospy.logdebug("ROS Image converted for processing")
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

        tic = time.time()
        boxes, confs, clss = self.trt_yolo.detect3(cv_img, self.conf_th)
        cv_img = self.vis.draw_bboxes(cv_img, boxes, confs, clss)
        toc = time.time()
        fps = 1.0 / (toc - tic)
        #print("Cam3: " + str(fps) + "\n")
        self.publisher(boxes, confs, clss, "Right Camera")
        """
            if self.show_img:
                cv_img = show_fps(cv_img, fps)
                cv2.imshow("Cam 3", cv_img)
                cv2.waitKey(1)
        """
        # converts back to ros_img type for publishing
        try:
            overlay_img = self.bridge.cv2_to_imgmsg(
                cv_img, encoding="bgr8")
            rospy.logdebug("CV Image converted for publishing")
            self.detection_image_publisher3.publish(overlay_img)
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

    def img_callback4(self, ros_img):
        """Continuously capture images from camera and do object detection """

        # converts from ros_img to cv_img for processing
        try:
            cv_img = self.bridge.imgmsg_to_cv2(
                ros_img, desired_encoding="bgr8")
            rospy.logdebug("ROS Image converted for processing")
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

        tic = time.time()
        boxes, confs, clss = self.trt_yolo.detect4(cv_img, self.conf_th)
        cv_img = self.vis.draw_bboxes(cv_img, boxes, confs, clss)
        toc = time.time()
        fps = 1.0 / (toc - tic)
        #print("Cam4: " + str(fps) + "\n")
        self.publisher(boxes, confs, clss, "Top Camera")
        """
            if self.show_img:
                cv_img = show_fps(cv_img, fps)
                cv2.imshow("Cam 3", cv_img)
                cv2.waitKey(1)
        """
        # converts back to ros_img type for publishing
        try:
            overlay_img = self.bridge.cv2_to_imgmsg(
                cv_img, encoding="bgr8")
            rospy.logdebug("CV Image converted for publishing")
            self.detection_image_publisher4.publish(overlay_img)
        except CvBridgeError as e:
            rospy.loginfo("Failed to convert image %s", str(e))

    def publisher(self, boxes, confs, clss, frame):
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
                detection.header.frame_id = frame  # change accordingly
                detection.results.id = clss[i]
                detection.results.score = confs[i]

                detection.bbox.center.x = boxes[i][0] + \
                    (boxes[i][2] - boxes[i][0])/2
                detection.bbox.center.y = boxes[i][1] + \
                    (boxes[i][3] - boxes[i][1])/2
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
