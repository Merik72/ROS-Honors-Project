// The purpose of this script is to teleport the goal away when the robot gets to it
using System.Collections;
using System.Collections.Generic;
using System.Linq.Expressions;
using UnityEngine;

[DefaultExecutionOrder(-1)]
public class Waypointing : MonoBehaviour
{
    public int index;
    private Vector3[] points = {
        new Vector3(-8.17000008f,-1.29999995f,13.96f),
        new Vector3(-5.98000002f,-1.29999995f,13.5100002f),
        new Vector3(-3.57999992f,-1.29999995f,13.1899996f),
        new Vector3(0.0500000007f,-1.29999995f,13.2700005f),
        new Vector3(-0.649999976f,-1.29999995f,10.7799997f),
        new Vector3(-2.68000007f,-1.29999995f,10.1400003f)
    };
    private void Awake()
    {
        index = 0;
        transform.localPosition = points[index];
    }

    public void RESTART_SCENE()
    {
        index = 0;
        transform.localPosition = points[index];
    }
    private void OnTriggerEnter(Collider other)
    {
        if (other.tag == "robot")
        {
            print(transform.root.name + " ball touched");
            transform.localPosition = points[++index];
            other.GetComponent<UpdatedWheelSubscriber>().BALL_TOUCHED();
        }

    }
}
