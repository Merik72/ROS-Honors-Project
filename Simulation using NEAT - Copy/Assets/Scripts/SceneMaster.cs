using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using Unity.VisualScripting;


public class SceneMaster : MonoBehaviour
{ 
    public static SceneMaster sceneMaster;
    public const string NODE_NAME = "/SceneMaster";
    public const string RESET_TOPIC = "/reset";
    public const string STARTUP_TOPIC = "/start";
    public State state;

    public GameObject visibleEnvironmentPrefab;
    public GameObject invisibleEnvironmentPrefab;

    public List<GameObject> environments = new List<GameObject>();

    ROSConnection ros_receiveCommand;
    
    void Start()
    {
        ROSConnection.GetOrCreateInstance().Subscribe<BoolMsg>(RESET_TOPIC, resetCallback);
        ROSConnection.GetOrCreateInstance().Subscribe<Int16Msg>(STARTUP_TOPIC, startupCallback);
    }
    private void Awake()
    {
        if(starting) startupCallback(numInstances);
    }

    public void startupCallback(Int16Msg numInstances)
    {
        print("message received");
        for (int i = 0; i < numInstances.data; i++)
        {
            if (i == 0)
            {
                environments.Add(Instantiate(visibleEnvironmentPrefab, new Vector3(0, i * 12f, 0), Quaternion.identity));
                environments[i].name = "env_" + i.ToString();
            }
            else
            {
                environments.Add(Instantiate(invisibleEnvironmentPrefab, new Vector3(0, i * 12f, 0), Quaternion.identity));
                environments[i].name = "env_" + i.ToString();
            }
        }
    }

    
    public bool starting = false;
    public int numInstances = 0;
    public void startupCallback(int numInstances)
    {
        starting = false;
        for (int i = 0; i < numInstances; i++)
        {
            if (i == 0)
            {
                environments.Add(Instantiate(visibleEnvironmentPrefab, new Vector3(0, i * 12f, 0), Quaternion.identity));
                environments[i].name = "env_" + i.ToString();
            }
            else
            {
                environments.Add(Instantiate(invisibleEnvironmentPrefab, new Vector3(0, i * 12f, 0), Quaternion.identity));
                environments[i].name = "env_" + i.ToString();
            }
        }
    }
     

    
    public void resetCallback(BoolMsg reset)
    {
        foreach (GameObject go in environments)
        {
            Transform tfdingo = go.transform.Find("dingo").Find("base_link").Find("chassis_link");
            GameObject godingo = tfdingo.gameObject;
            godingo.GetComponent<UpdatedWheelSubscriber>().RESTART_SCENE(); 
            print("reset " + go.name + " belonging to " + go.transform.root.name);

            Transform tflineRenderer = tfdingo.Find("PathTracer");
            GameObject goPathTracer = tflineRenderer.gameObject;
            goPathTracer.GetComponent<pathTracer>().RESTART_SCENE();
            /*
            Transform tfTarget = go.transform.Find("Flattened reshaped scan").Find("TARGET");
            GameObject goTarget = tfTarget.gameObject;
            goTarget.GetComponent<Waypointing>().RESTART_SCENE();
            */
        }
    }

    
    public bool reset = false;
    public void resetCallback()
    {
        reset = false;
        foreach (GameObject go in environments)
        {
            Transform tfdingo = go.transform.Find("dingo").Find("base_link").Find("chassis_link");
            GameObject godingo = tfdingo.gameObject;
            godingo.GetComponent<UpdatedWheelSubscriber>().RESTART_SCENE(); // find how to make this work
            // godingo.GetComponent<UpdatedWheelSubscriber>();
            print("reset " + go.name + " belonging to " + go.transform.root.name);
            Transform tflineRenderer = tfdingo.Find("PathTracer");
            GameObject goPathTracer = tflineRenderer.gameObject;
            goPathTracer.GetComponent<pathTracer>().RESTART_SCENE();

            Transform tfTarget = go.transform.Find("Flattened reshaped scan").Find("TARGET");
            GameObject goTarget = tfTarget.gameObject;
            goTarget.GetComponent<Waypointing>().RESTART_SCENE();
        }
    }
     

    // Camera control is probably more effort than it's worth
}
