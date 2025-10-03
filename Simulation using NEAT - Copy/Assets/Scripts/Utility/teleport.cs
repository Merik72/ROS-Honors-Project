using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UIElements;

[DefaultExecutionOrder(-1)]
public class teleport : MonoBehaviour
{
    public static int iter = 0;
    // Start is called before the first frame update
    void OnEnable()
    {
        string name = transform.root.name;
        Vector3 startingLocation;
        Quaternion startingRotation;
        transform.root.Find("51 random spawn points").GetChild(++iter).GetPositionAndRotation(out startingLocation, out startingRotation);
        transform.SetPositionAndRotation(new Vector3(startingLocation.x, transform.position.y, startingLocation.z), startingRotation);
    }
}
