using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using WheelSpin = RosMessageTypes.UnityRoboticsDemo.WheelSpinMsg;
//using RosVelo = RosMessageTypes.UnityRoboticsDemo.Velocity;

public class WheelSubscriber : MonoBehaviour
{
    public ArticulationBody leftWheel;
    public ArticulationBody rightWheel;
    public float leftGoal = 0;
    public float rightGoal = 0;
    public int moveSpeed = 5;
    public int dashSpeed = 7;
    public int normalSpeed = 5;

    void Start()
    {
        ROSConnection.GetOrCreateInstance().Subscribe<WheelSpin>("movement", MoveMent);
    }

    void MoveMent(WheelSpin moveMessage)
    {
        leftGoal += moveMessage.left;
        rightGoal += moveMessage.right;
        //leftWheel.AddTorque(transform.right*moveMessage.left);
        //rightWheel.AddTorque(transform.forward * moveMessage.left);
        //leftWheel.SetDriveTargetVelocity(ArticulationDriveAxis.X, moveMessage.left);
        //rightWheel.SetDriveTargetVelocity(ArticulationDriveAxis.X, moveMessage.right);
        leftWheel.SetDriveTarget(ArticulationDriveAxis.X, leftGoal);
        rightWheel.SetDriveTarget(ArticulationDriveAxis.X, rightGoal);
        //Debug.Log("Received text: " + moveMessage);
        //leftWheel.SetDriveTarget(ArticulationDriveAxis.X, moveMessage.left);
        //rightWheel.SetDriveTarget(ArticulationDriveAxis.X, moveMessage.right);

        //bool w, a, s, d;
        //w = Input.GetKey("w");
        //a = Input.GetKey("a");
        //s = Input.GetKey("s");
        //d = Input.GetKey("d");


        //if (w || s)
        //{
        //    moveSpeed = dashSpeed;
        //    if (w)
        //    {
        //        leftGoal += moveSpeed;
        //        rightGoal += moveSpeed;
        //    }
        //    if (s)
        //    {
        //        leftGoal -= moveSpeed;
        //        rightGoal -= moveSpeed;
        //    }
        //    if (a)
        //    {
        //        rightGoal += moveSpeed;
        //    }
        //    if (d)
        //    {
        //        leftGoal += moveSpeed;
        //    }
        //}
        //else
        //{
        //    moveSpeed = normalSpeed;
        //    if (a)
        //    {
        //        leftGoal -= moveSpeed;
        //        rightGoal += moveSpeed;
        //    }
        //    if (d)
        //    {
        //        leftGoal += moveSpeed;
        //        rightGoal -= moveSpeed;
        //    }
        //}
        //leftWheel.SetDriveTarget(ArticulationDriveAxis.X, leftGoal);
        //rightWheel.SetDriveTarget(ArticulationDriveAxis.X, rightGoal);
    }
}