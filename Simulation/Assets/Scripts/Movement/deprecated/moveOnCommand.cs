using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// These are super important.////////////////////////////////////////////////////////////////////////
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Geometry;


using System.IO;
using UnityEngine.UIElements;
using Unity.VisualScripting;
// Added for temporary implementation of shutdown command
// using UnityColor = RosMessageTypes.UnityRoboticsDemo.UnityColorMsg;

// Current theory of  issue with the robot - Time is too short, and it doesn't get rewarded enough for moving directly away from the origin.
// Observation: Dingo will drive backwards towards the wall, slightly turning over and over to avoid it, inching towards it,
// getting as close as it can before time runs out

// Proposed Solution: On top of Delta displacement, also reward total displacement again in some way.
public class MoveOnCommand : MonoBehaviour
{
    // MESSAGING //
    public string WHEEL_TOPIC = "/vel/wheels";
    // ROBOT CONTROL //
    public ArticulationBody chassisControl, leftWheel, rightWheel;
    public float rightVelocity;
    public float leftVelocity;
    public float speed = 1000f;
    public MeshCollider chassisCollider;

    private void Start()
    {
        ROSConnection.GetOrCreateInstance().Subscribe<TwistMsg>(WHEEL_TOPIC, listen);// make this dingo specific
    }

    void listen(TwistMsg msg)
    {
        rightVelocity = (float)msg.linear.x * speed + -(float)(msg.angular.z) * speed;
        leftVelocity = (float)(msg.angular.z) * speed + (float)msg.linear.x * speed;

        rightVelocity = (rightVelocity > speed) ? speed : (rightVelocity < -speed) ? -speed : rightVelocity;
        leftVelocity = (leftVelocity > speed) ? speed : (leftVelocity < -speed) ? -speed : leftVelocity;
    }

    // public int publishTime = 0;
    private void FixedUpdate()
    {
        leftWheel.SetDriveTargetVelocity(ArticulationDriveAxis.X, leftVelocity);
        rightWheel.SetDriveTargetVelocity(ArticulationDriveAxis.X, rightVelocity);
    }
}