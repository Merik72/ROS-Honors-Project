// The purpose of this script is to teleport the goal away when the robot gets to it
using System.Collections;
using System.Collections.Generic;
using System.Linq.Expressions;
using UnityEngine;
/*
public class Waypointing : MonoBehaviour
{
    public int index;
    private Vector3[] points = {
        new Vector3(-5,-1.3f,13.5f),
        new Vector3(-1.4f,-1.3f,13.32f),
        new Vector3(0,-1.3f,13),
        new Vector3(0,-1.3f,11),
        new Vector3(-1.34f, -1.3f, 10.13f),
        new Vector3(-3.46f,-1.3f,10)
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
            print(transform.root.name + " ball touched" );
            transform.localPosition = points[++index];
            other.GetComponent<UpdatedWheelSubscriber>().ballsTouched += 1;
        }

    }
}

 */