using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using RosMessageTypes.Std;
using Unity.Robotics.ROSTCPConnector;

[RequireComponent(typeof(LineRenderer))]
public class pathTracer : MonoBehaviour
{
    public string DISTANCE_TOPIC;
    ROSConnection publishDistance;
    private LineRenderer _line;

    private bool _drawing = true;
    private ArticulationBody _articulationBody;


    // Start is called before the first frame update
    void Start()
    {
        DISTANCE_TOPIC = /*transform.root.name + */ "/distance";
        publishDistance = ROSConnection.GetOrCreateInstance();
        publishDistance.RegisterPublisher<Float32Msg>(DISTANCE_TOPIC);



        _line = GetComponent<LineRenderer>();
        _line.positionCount = 1;

        _line.SetPosition(0, transform.position);

        _articulationBody = GetComponent<ArticulationBody>();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if (_drawing && Time.frameCount % 10 == 0)
        {
            _line.positionCount++;
            Vector3 position = transform.position;
            position.y += 0.1f;
            _line.SetPosition(_line.positionCount - 1, position);
            if (_articulationBody != null)
            {
                if (!_articulationBody.enabled)
                {
                    _drawing = false;
                    _articulationBody = null;
                }
            }
        }

    }
    public void RESTART_SCENE()
    {
        Vector3[] positions = new Vector3[_line.positionCount];
        _line.GetPositions(positions);
        float distanceApproximate = 0;
        for (int i = 0; i < positions.Length-1; i++)
        {
            distanceApproximate += Vector3.Distance(positions[i], positions[i + 1]);
        }
        print(distanceApproximate);
        Float32Msg distMsg = new Float32Msg();
        distMsg.data = distanceApproximate;
        publishDistance.Publish(DISTANCE_TOPIC, distMsg);
        distanceApproximate = 0;
        _line.positionCount = 0;
    }
}
