;; Auto-generated. Do not edit!


(when (boundp 'unity_robotics_demo_msgs::Twist)
  (if (not (find-package "UNITY_ROBOTICS_DEMO_MSGS"))
    (make-package "UNITY_ROBOTICS_DEMO_MSGS"))
  (shadow 'Twist (find-package "UNITY_ROBOTICS_DEMO_MSGS")))
(unless (find-package "UNITY_ROBOTICS_DEMO_MSGS::TWIST")
  (make-package "UNITY_ROBOTICS_DEMO_MSGS::TWIST"))

(in-package "ROS")
;;//! \htmlinclude Twist.msg.html


(defclass unity_robotics_demo_msgs::Twist
  :super ros::object
  :slots (_linear _angular ))

(defmethod unity_robotics_demo_msgs::Twist
  (:init
   (&key
    ((:linear __linear) (instance unity_robotics_demo_msgs::Vector3 :init))
    ((:angular __angular) (instance unity_robotics_demo_msgs::Vector3 :init))
    )
   (send-super :init)
   (setq _linear __linear)
   (setq _angular __angular)
   self)
  (:linear
   (&rest __linear)
   (if (keywordp (car __linear))
       (send* _linear __linear)
     (progn
       (if __linear (setq _linear (car __linear)))
       _linear)))
  (:angular
   (&rest __angular)
   (if (keywordp (car __angular))
       (send* _angular __angular)
     (progn
       (if __angular (setq _angular (car __angular)))
       _angular)))
  (:serialization-length
   ()
   (+
    ;; unity_robotics_demo_msgs/Vector3 _linear
    (send _linear :serialization-length)
    ;; unity_robotics_demo_msgs/Vector3 _angular
    (send _angular :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; unity_robotics_demo_msgs/Vector3 _linear
       (send _linear :serialize s)
     ;; unity_robotics_demo_msgs/Vector3 _angular
       (send _angular :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; unity_robotics_demo_msgs/Vector3 _linear
     (send _linear :deserialize buf ptr-) (incf ptr- (send _linear :serialization-length))
   ;; unity_robotics_demo_msgs/Vector3 _angular
     (send _angular :deserialize buf ptr-) (incf ptr- (send _angular :serialization-length))
   ;;
   self)
  )

(setf (get unity_robotics_demo_msgs::Twist :md5sum-) "9f195f881246fdfa2798d1d3eebca84a")
(setf (get unity_robotics_demo_msgs::Twist :datatype-) "unity_robotics_demo_msgs/Twist")
(setf (get unity_robotics_demo_msgs::Twist :definition-)
      "Vector3 linear
Vector3 angular
================================================================================
MSG: unity_robotics_demo_msgs/Vector3
float64 x
float64 y
float64 z
")



(provide :unity_robotics_demo_msgs/Twist "9f195f881246fdfa2798d1d3eebca84a")


