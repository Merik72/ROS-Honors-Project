// This is lidar sensor code by Johnathan Platt
using System;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;
using RosMessageTypes.BuiltinInterfaces;
using Unity.Robotics.Core;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
// using UnityEngine.Serialization;
using Unity.Collections;
using Unity.Jobs;

public class VelodyneLidarSensor : MonoBehaviour
{
    public string topic = "/mid/points";
    public double PublishPeriodSeconds = 0.1;
    public float RangeMetersMin = 0.9f;
    public float RangeMetersMax = 130;
    public float noise = 0.008f;
    public float ScanAngleStartDegrees = 180;
    public float ScanAngleEndDegrees = -180;
    public float pitchAngleStartDegrees = -15;
    public float pitchAngleEndDegrees = 15;
    public int lasers = 16;
    public int samples = 1875;
    public string FrameId = "velodyne";

    float m_CurrentScanAngleStart;
    float m_CurrentScanAngleEnd;
    ROSConnection m_Ros;
    double m_TimeNextScanSeconds = -1;
    NativeArray<RaycastHit> results;
    NativeArray<RaycastCommand> commands;
    JobHandle scanHandle;
    float pitchIncrement;
    float yawIncrement;
    byte[] points;

    bool isScanning = false;
    double m_TimeLastScanBeganSeconds = -1;

    protected virtual void Start()
    {
        topic = transform.root.name + topic;

        results = new NativeArray<RaycastHit>(lasers * samples, Allocator.Persistent);
        commands = new NativeArray<RaycastCommand>(lasers * samples, Allocator.Persistent);
        points = new byte[lasers * samples * 22];
        pitchIncrement = (pitchAngleEndDegrees - pitchAngleStartDegrees) / (Math.Max(1, lasers - 1));
        yawIncrement = (ScanAngleEndDegrees - ScanAngleStartDegrees) / (Math.Max(1, samples - 1));

        m_Ros = ROSConnection.GetOrCreateInstance();
        m_Ros.RegisterPublisher<PointCloud2Msg>(topic);

        m_CurrentScanAngleStart = ScanAngleStartDegrees;
        m_CurrentScanAngleEnd = ScanAngleEndDegrees;

        m_TimeNextScanSeconds = Clock.Now + PublishPeriodSeconds;
    }

    public void Update()
    {
        if (isScanning && scanHandle.IsCompleted)
        {
            scanHandle.Complete();
            EndScan();
            isScanning = false;
        }
        if (!isScanning)
        {
            if (Clock.NowTimeInSeconds < m_TimeNextScanSeconds)
            {
                return;
            }
            BeginScan();
        }
    }

    void OnDestroy()
    {
        if (results.IsCreated) results.Dispose();
        if (commands.IsCreated) commands.Dispose();
    }

    void BeginScan()
    {
        isScanning = true;
        m_TimeLastScanBeganSeconds = Clock.Now;
        m_TimeNextScanSeconds = m_TimeLastScanBeganSeconds + PublishPeriodSeconds;

        var pitchSensorDegrees = pitchAngleStartDegrees;
        for (int i = 0; i < lasers; i++)
        {
            var pitchVector = Quaternion.AngleAxis(pitchSensorDegrees, transform.right) * transform.forward;

            var yawSensorDegrees = ScanAngleStartDegrees;
            for (int j = 0; j < samples; j++)
            {
                var directionVector = Quaternion.AngleAxis(yawSensorDegrees, transform.up) * pitchVector;
                commands[i * samples + j] = new RaycastCommand(transform.position, directionVector, RangeMetersMax);
                // Debug.DrawRay(transform.position,directionVector,Color.red,1);
                yawSensorDegrees += yawIncrement;
            }
            pitchSensorDegrees += pitchIncrement;
        }

        scanHandle = RaycastCommand.ScheduleBatch(commands, results, 1, default(JobHandle));
    }

    // TODO : Find a faster way to make a random number for fast random
    //      : Check if this gaussian noise method is actually fast enough for realtime
    //public float gaussianNoise()
    //{
    //    FastRandom _rng = new FastRandom();
    //    double? _spareValue = null;
    //    double NextDouble()
    //    {
    //        if (null != _spareValue)
    //        {
    //            double tmp = _spareValue.Value;
    //            _spareValue = null;
    //            return tmp;
    //        }

    //        // Generate two new gaussian values.
    //        double x, y, sqr;

    //        // We need a non-zero random point inside the unit circle.
    //        do
    //        {
    //            x = 2.0 * _rng.NextDouble() - 1.0;
    //            y = 2.0 * _rng.NextDouble() - 1.0;
    //            sqr = x * x + y * y;
    //        }
    //        while (sqr > 1.0 || sqr == 0);

    //        // Make the Box-Muller transformation.
    //        double fac = Math.Sqrt(-2.0 * Math.Log(sqr) / sqr);

    //        _spareValue = x * fac;
    //        return y * fac;
    //    }
    //    return (float)(noise * NextDouble());
    //}

    public void EndScan()
    {
        int pointCount = 0;
        Array.Resize(ref points,lasers*samples*22);
        for(int i = 0; i < results.Length; i++)
        {
            /*
            //yoyoyoyosef on stack exchange
                System.Random rand = new System.Random(); //reuse this if you are generating many
                double u1 = 1.0 - rand.NextDouble(); //uniform(0,1] random doubles
                double u2 = 1.0 - rand.NextDouble();
                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
                double randNormal = noise * randStdNormal; //random normal(mean,stdDev^2)
            //end yoyoyoyosef on stack exchange 
             */

            float distance = results[i].distance; // + gaussianNoise();
            if(results[i].collider != null && 
               results[i].distance < RangeMetersMax &&
               results[i].distance > RangeMetersMin)
            {
                int laserNumber = i / samples;
                int sampleNumber = i % samples;
                // Negative to account for different coordinate systems between Unity and ROS
                double pitch = -(laserNumber*pitchIncrement + pitchAngleStartDegrees) * Math.PI / 180;
                double yaw = -(sampleNumber*yawIncrement + ScanAngleStartDegrees) * Math.PI / 180;
                double range = results[i].distance;
                
                float x = (float)(range * Math.Cos(pitch) * Math.Cos(yaw));
                float y = (float)(range * Math.Cos(pitch) * Math.Sin(yaw));
                float z = (float)(range * Math.Sin(pitch));
                float intensity = 0;
                UInt16 ring = (UInt16)laserNumber;
                float time = 0;
                
                // debug
                /*
                if(i == 0)
                {
                    Debug.Log("x, y, z: (" + x + ", " + y + ", " + z + ")");
                }
                */

                Array.Copy(BitConverter.GetBytes(x),0,points,pointCount*22,4);
                Array.Copy(BitConverter.GetBytes(y),0,points,pointCount*22+4,4);
                Array.Copy(BitConverter.GetBytes(z),0,points,pointCount*22+8,4);
                Array.Copy(BitConverter.GetBytes(intensity),0,points,pointCount*22+12,4);
                Array.Copy(BitConverter.GetBytes(ring),0,points,pointCount*22+16,2);
                Array.Copy(BitConverter.GetBytes(time),0,points,pointCount*22+18,4);

                pointCount++;
            }
        }
        Array.Resize(ref points,pointCount*22);
        if (pointCount == 0)
        {
            Debug.LogWarning($"Found no valid ranges");
        }

        var timestamp = new TimeStamp(Clock.time);
        // Invert the angle ranges when going from Unity to ROS
        var angleStartRos = -m_CurrentScanAngleStart * Mathf.Deg2Rad;
        var angleEndRos = -m_CurrentScanAngleEnd * Mathf.Deg2Rad;
        if (angleStartRos > angleEndRos)
        {
            Debug.LogWarning("LaserScan was performed in a clockwise direction but ROS expects a counter-clockwise scan, flipping the ranges...");
            var temp = angleEndRos;
            angleEndRos = angleStartRos;
            angleStartRos = temp;
            Array.Reverse(points);
        }

        var msg = new PointCloud2Msg
        {
            header = new HeaderMsg
            {
                frame_id = FrameId,
                stamp = new TimeMsg((uint)timestamp.Seconds,timestamp.NanoSeconds)
            },
            height = 1,
            width = (uint)pointCount,
            fields = new PointFieldMsg[]
            {
                new PointFieldMsg("x",0,PointFieldMsg.FLOAT32,1),
                new PointFieldMsg("y",4,PointFieldMsg.FLOAT32,1),
                new PointFieldMsg("z",8,PointFieldMsg.FLOAT32,1),
                new PointFieldMsg("intensity",12,PointFieldMsg.FLOAT32,1),
                new PointFieldMsg("ring",16,PointFieldMsg.UINT16,1),
                new PointFieldMsg("time",18,PointFieldMsg.FLOAT32,1)
            },
            is_bigendian = false,
            point_step = 22,
            row_step = (uint)pointCount * 22,
            data = points,
            is_dense = true
        };
        
        m_Ros.Publish(topic, msg);

        isScanning = false;
        var now = (float)Clock.time;
        if (now > m_TimeNextScanSeconds)
        {
            Debug.LogWarning($"Failed to complete scan started at {m_TimeLastScanBeganSeconds:F} before next scan was " +
                             $"scheduled to start: {m_TimeNextScanSeconds:F}, rescheduling to now ({now:F})");
            m_TimeNextScanSeconds = now;
        }
    }
}