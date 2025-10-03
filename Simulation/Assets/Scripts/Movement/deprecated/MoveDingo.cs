using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveCar : MonoBehaviour
{
    public float moveSpeed = 10;
  
    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

        float x = Input.GetAxisRaw("Horizontal");
        float z = Input.GetAxisRaw("Vertical");
        Vector3 movement = new Vector3(x, 0, z);

        transform.Translate(movement * moveSpeed * Time.deltaTime);

        if (Input.GetKeyDown(KeyCode.UpArrow))
        {
            Debug.Log("just click key up");
        }
        if (Input.GetKeyDown(KeyCode.DownArrow))
        {
            Debug.Log("just click key up");
        }
    }
}



//using System.Collections;
//using System.Collections.Generic;
//using UnityEngine;

//public class RobotController : MonoBehaviour
//{
//    public float moveSpeed = 3f;
//    public float rotationSpeed = 100f;

//    private Rigidbody rb;

//    void Start()
//    {
//        rb = GetComponent<Rigidbody>();
//    }

//    void Update()
//    {
//        float moveInput = Input.GetAxis("Vertical");
//        float rotationInput = Input.GetAxis("Horizontal");

//        Vector3 movement = transform.forward * moveInput * moveSpeed * Time.deltaTime;
//        Quaternion rotation = Quaternion.Euler(0, rotationInput * rotationSpeed * Time.deltaTime, 0);

//        rb.MovePosition(rb.position + movement);
//        rb.MoveRotation(rb.rotation * rotation);
//    }
//}
