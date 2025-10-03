using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class moveWheels : MonoBehaviour
{
    public float moveSpeed = 2.5f;
    public float normalSpeed = 2.5f;
    public float dashSpeed = 7f;
    public ArticulationBody leftWheel;
    public ArticulationBody rightWheel;
    public float leftGoal = 0, rightGoal = 0;
    public float rotation = 0;
    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void FixedUpdate()
    {

        //float horz = Input.GetAxisRaw("Horizontal");
        //float vert = Input.GetAxisRaw("Vertical");
        //leftGoal += vert * moveSpeed;
        //rightGoal += vert * moveSpeed;
        //Debug.Log(horz);
        //if(horz == -1)
        //{
        //    leftGoal += moveSpeed / rotationModifier;
        //}
        //if (Input.GetKey("d"))
        //{
        //    rightGoal += moveSpeed / rotationModifier;
        //}
        //if(vert == 1)
        //{
        //    leftGoal += moveSpeed;
        //    rightGoal += moveSpeed;
        //}
        //leftWheel.SetDriveTarget(ArticulationDriveAxis.X, leftGoal);
        //rightWheel.SetDriveTarget(ArticulationDriveAxis.X, rightGoal);

        bool q, e, a, d;
        q = Input.GetKey("q");
        e = Input.GetKey("e");

        a = Input.GetKey("a");
        d = Input.GetKey("d");


        if ((q && e)|| ( a&&d))
        {
            moveSpeed = dashSpeed;
        }
        else
        {
            moveSpeed = normalSpeed;
        }
        if (q && d)
        {
            leftGoal += moveSpeed;
            rightGoal -= moveSpeed;
        }
        else if (e && a)
        {
            leftGoal -= moveSpeed;
            rightGoal += moveSpeed;
        }
        else
        {
            if (q)
            {
                leftGoal += moveSpeed;
            }
            if (a)
            {
                leftGoal -= moveSpeed;
            }
            if (e)
            {
                rightGoal += moveSpeed;
            }
            if (d)
            {
                rightGoal -= moveSpeed;
            }
        }
        
        leftWheel.SetDriveTarget(ArticulationDriveAxis.X, leftGoal);
        rightWheel.SetDriveTarget(ArticulationDriveAxis.X, rightGoal);

    }
}