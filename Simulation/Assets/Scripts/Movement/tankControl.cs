using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public class twistMuxRecreate : MonoBehaviour
{
    //public float normalSpeed = 4f;
    public float speed = 1300f;
    public ArticulationBody leftWheel;
    public ArticulationBody rightWheel;
    public ArticulationBody chassis;
    public float leftGoal = 0, rightGoal = 0;
    public float rotation = 0;
    public Vector3 linear;
    public Vector3 angular;
    public bool w, a, s, d;
    public int driveStyle; // 0 is normal, 1 is vectoring
    public float turn;
    public float move;
    
    public float leftVelocityLinear = 0;
    public float rightVelocityLinear = 0;
    public float leftVelocityAngular = 0;
    public float rightVelocityAngular = 0;
    public float rightVelocity;
    public float leftVelocity;

    private void Awake()
    {
        leftWheel = transform.Find("left_wheel_link").gameObject.GetComponent<ArticulationBody>();
        rightWheel = transform.Find("right_wheel_link").gameObject.GetComponent<ArticulationBody>();
        chassis = gameObject.GetComponent<ArticulationBody>();
    }
    void FixedUpdate()
    {
        
        rightVelocity = ((float)Input.GetAxis("Vertical") * speed + -(float)(Input.GetAxis("Horizontal")) * speed) ;
        leftVelocity = ((float)(Input.GetAxis("Vertical")) * speed + (float)Input.GetAxis("Horizontal") * speed) ;

        // rightVelocity = (rightVelocity > speed) ? speed : (rightVelocity < -speed) ? -speed : rightVelocity;
        // leftVelocity = (leftVelocity > speed) ? speed : (leftVelocity < -speed) ? -speed : leftVelocity;
        if (Math.Abs(rightVelocity) > speed)
        {
            if (rightVelocity > 0f)
            {
                rightVelocity = speed;
            }
            else if (rightVelocity < 0f)
            {
                rightVelocity = -speed;
            }
        }
        if (Math.Abs(leftVelocity) > speed)
        {
            if (leftVelocity > 0f)
            {
                leftVelocity = speed;
            }
            else if (leftVelocity < 0f)
            {
                leftVelocity = -speed;
            }
        }

        if (chassis.velocity.magnitude > 1)
        {

            if ((rightVelocity) < -speed / 2)
            {

                rightVelocity = -speed / 2;

            }
            if ((leftVelocity) < -speed / 2)
            {

                leftVelocity = -speed / 2;

            }
        }
        print("SPEED"+chassis.velocity.magnitude);
        if (chassis.angularVelocity.magnitude > 0.5f)
        {
            print("NIPS"+chassis.angularVelocity.magnitude);
            if (Math.Abs(rightVelocity) > speed/2)
            {
                if (rightVelocity > 0f)
                {
                    rightVelocity = speed / 2;
                }
                else if (rightVelocity < 0f)
                {
                    rightVelocity = -speed / 2;
                }
            }
            if (Math.Abs(leftVelocity) > speed / 2)
            {
                if (leftVelocity > 0f)
                {
                    leftVelocity = speed / 2;
                }
                else if (leftVelocity < 0f)
                {
                    leftVelocity = -speed / 2;
                }
            }
        }
        leftWheel.SetDriveTargetVelocity(ArticulationDriveAxis.X, leftVelocity);
        rightWheel.SetDriveTargetVelocity(ArticulationDriveAxis.X, rightVelocity);
        // leftWheel.SetDriveTarget(ArticulationDriveAxis.X, leftGoal);
        // rightWheel.SetDriveTarget(ArticulationDriveAxis.X, rightGoal);

        /*
        if (w)
        {
            moveSpeed = dashSpeed;
            if (w)
            {
                leftGoal += moveSpeed;
                rightGoal += moveSpeed;
            }
            if (s)
            {
                leftGoal -= moveSpeed;
                rightGoal -= moveSpeed;
            }
            if (a)
            {
                rightGoal += moveSpeed;
            }
            if (d)
            {
                leftGoal += moveSpeed;
            }
        }
        else
        {
            moveSpeed = normalSpeed;
            if (a)
            {
                leftGoal -= moveSpeed;
                rightGoal += moveSpeed;
            }
            if (d)
            {
                leftGoal += moveSpeed;
                rightGoal -= moveSpeed;
            }
        }
        */

    }
}