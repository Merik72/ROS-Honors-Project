
(cl:in-package :asdf)

(defsystem "unity_robotics_demo_msgs-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "PosRot" :depends-on ("_package_PosRot"))
    (:file "_package_PosRot" :depends-on ("_package"))
    (:file "Twist" :depends-on ("_package_Twist"))
    (:file "_package_Twist" :depends-on ("_package"))
    (:file "UnityColor" :depends-on ("_package_UnityColor"))
    (:file "_package_UnityColor" :depends-on ("_package"))
    (:file "Vector3" :depends-on ("_package_Vector3"))
    (:file "_package_Vector3" :depends-on ("_package"))
    (:file "Velocity" :depends-on ("_package_Velocity"))
    (:file "_package_Velocity" :depends-on ("_package"))
  ))