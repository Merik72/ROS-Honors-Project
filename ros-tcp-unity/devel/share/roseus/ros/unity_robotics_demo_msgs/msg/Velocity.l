;; Auto-generated. Do not edit!


(when (boundp 'unity_robotics_demo_msgs::Velocity)
  (if (not (find-package "UNITY_ROBOTICS_DEMO_MSGS"))
    (make-package "UNITY_ROBOTICS_DEMO_MSGS"))
  (shadow 'Velocity (find-package "UNITY_ROBOTICS_DEMO_MSGS")))
(unless (find-package "UNITY_ROBOTICS_DEMO_MSGS::VELOCITY")
  (make-package "UNITY_ROBOTICS_DEMO_MSGS::VELOCITY"))

(in-package "ROS")
;;//! \htmlinclude Velocity.msg.html


(defclass unity_robotics_demo_msgs::Velocity
  :super ros::object
  :slots ())

(defmethod unity_robotics_demo_msgs::Velocity
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(setf (get unity_robotics_demo_msgs::Velocity :md5sum-) "d41d8cd98f00b204e9800998ecf8427e")
(setf (get unity_robotics_demo_msgs::Velocity :datatype-) "unity_robotics_demo_msgs/Velocity")
(setf (get unity_robotics_demo_msgs::Velocity :definition-)
      "
")



(provide :unity_robotics_demo_msgs/Velocity "d41d8cd98f00b204e9800998ecf8427e")


