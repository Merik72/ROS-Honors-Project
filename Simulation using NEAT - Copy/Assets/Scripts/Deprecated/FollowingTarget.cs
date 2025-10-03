using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FollowingTarget : MonoBehaviour
{
    // Start is called before the first frame update
   
    public ArticulationBody articulationBody;
    private float moveSpeed = 5.0f;
    float deltaMoveSpeed;
    float accel;
    private Quaternion localQuaternion;
    private Vector3 localPostion ;
    bool temp = false;
    float y;
    public float x;


    void Start()
    {
        // x = 1.511003f;
    }


    // Update is called once per frame
    void Update()
    {
        float leftrightkey = Input.GetAxisRaw("Horizontal");
        float upDownKey = Input.GetAxisRaw("Vertical");
        if (leftrightkey == 1| leftrightkey == -1)
        {
            y = y + leftrightkey* Time.deltaTime * 40;
            localQuaternion = Quaternion.Euler(0, y, 0);
        }
         transform.rotation = localQuaternion;

        // localPostion += new Vector3(upDownKey, 0, 0) * moveSpeed * Time.deltaTime;
        // Debug.Log("postion: " + transform.position);

        //Debug.Log(localPostion); 
       if (upDownKey == 1 || upDownKey == -1)
        {
            temp = true;

            //x = upDownKey * moveSpeed * 0.2f *Time.deltaTime;
            x=Mathf.SmoothStep(0, 500f, Time.deltaTime);
            localPostion += upDownKey*x * transform.forward;
        }
        if (upDownKey == 0) x = 0;
        Debug.Log(localPostion);
        if(temp) transform.SetPositionAndRotation(localPostion, localQuaternion);
        localPostion = transform.position;

        // Articulationbody needs to be done on wheels/joints that can move
        // articulationBody.AddForce(transform.forward * upDownKey * Time.deltaTime);
        // move in the forwards direction when press forwards on the keyboard


        // 0.5511003 - 1.334488  8.797976
        //articulationBody.TeleportRoot(pos, localQuaternion);
    }
}
