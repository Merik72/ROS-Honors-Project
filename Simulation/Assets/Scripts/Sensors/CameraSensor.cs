// This is camera code by Johnathan Platt
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Serialization;
using Unity.Robotics.Core;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;
using RosMessageTypes.BuiltinInterfaces;

public class CameraSensor : MonoBehaviour
{
    public string topicInfo;
    public string topicImage;
    public GameObject cameraGO;
    public Camera cameraObject;
    [FormerlySerializedAs("TimeBetweenScansSeconds")]
    public float tx = 0;
    public int width = 320;
    public int height = 240;
    public float gaussianNoiseMean = 0;
    public float gaussianNoiseStddev = 0.007f;
    private Material noiseMaterial;
    public double PublishPeriodSeconds = 0.1;

    public string FrameId = "base_scan";
    private const int isBigEndian = 0;
    private uint step;

    ROSConnection m_Ros;
    double m_TimeNextImageSeconds = -1;

    bool isCapturing = false;
    double m_TimeLastImageSeconds = -1;

    private RenderTexture renderTexture;
    Texture2D mainCameraTexture;

    int numImages = 0;

    void Awake ()
    {
        cameraGO = RecursiveFindChild(transform, "Camera1").gameObject;
        cameraObject = cameraGO.GetComponent<Camera>();
        noiseMaterial = new Material( Shader.Find("Custom/GaussianNoise") );
        noiseMaterial.SetFloat("_mean", gaussianNoiseMean);
        noiseMaterial.SetFloat("_stddev", gaussianNoiseStddev);
    }

    protected virtual void Start()
    {
        topicImage = transform.root.name + "/image_raw_1";
        //topicImage = transform.root.name + "/camera/rgb/image_raw";
        topicInfo = transform.root.name + "/camera/rgb/camera_info";
        m_Ros = ROSConnection.GetOrCreateInstance();
        m_Ros.RegisterPublisher<ImageMsg>(topicImage);
        m_Ros.RegisterPublisher<CameraInfoMsg>(topicInfo);
        
        // Render texture 
        step = (uint)width*3;
        renderTexture = new RenderTexture(width, height, 24);
       // renderTexture.Create();

        m_TimeNextImageSeconds = Clock.Now + PublishPeriodSeconds;
    }

    Transform RecursiveFindChild(Transform parent, string childName)
    {
        foreach (Transform child in parent)
        {
            if (child.name == childName)
            {
                return child;
            }
            else
            {
                Transform found = RecursiveFindChild(child, childName);
                if (found != null)
                {
                    //print(childName + " found");
                    return found;
                }
            }
        }
        return null;
    }

    private void sendCameraInfo()
    {
        uint imageHeight = (uint)renderTexture.height;
        uint imageWidth = (uint)renderTexture.width;

        string distortionModel = "plumb_bob";
        // double[] D = { -0.002326204753968315, -0.0008776349103417776, -5.2028236482803694e-05, -0.00019331827310204024, 0.0 };
        // double[] K = { 784.4392697041746, 0.0, 510.8221404889899, 0.0, 784.3088270414429, 383.27986235231697, 0.0, 0.0, 1.0 };
        // double[] R = { 0.9982171337271148, -0.03921084429664997, -0.0450007069185274, 0.03916260188582239, 0.9992309415097452, -0.0019534950935547126, 0.045042696934751894, 0.00018766750340733012, 0.998985045050101 };
        // double[] P = { 857.5183268530548, 0.0, 554.7425727844238, tx, 0.0, 857.5183268530548, 383.94894790649414, 0.0, 0.0, 0.0, 1.0, 0.0 };
        // tx = -100 for right

        // From Gazebo
        double[] D = { 0.0, 0.0, 0.0, 0.0, 0.0 };
        double[] K = { 788.4085367665094, 0.0, 512.5, 0.0, 788.4085367665094, 384.5, 0.0, 0.0, 1.0 };
        double[] R = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
        double[] P = { 788.4085367665094, 0.0, 512.5, tx, 0.0, 788.4085367665094, 384.5, 0.0, 0.0, 0.0, 1.0, 0.0 };
        // tx = -94.60902441198112 for right

        var timestamp = new TimeStamp(Clock.time);
        var cameraInfoMsg = new CameraInfoMsg {
            header = new HeaderMsg
            {
                frame_id = FrameId,
                stamp = new TimeMsg
                {
                    sec = (uint)timestamp.Seconds,
                    nanosec = timestamp.NanoSeconds,
                }
            },
            width = imageWidth,
            height = imageHeight,
            distortion_model = distortionModel,
            d = D,
            k = K,
            r = R,
            p = P
        };

        m_Ros.Publish(topicInfo, cameraInfoMsg);
    }

    private byte[] CaptureScreenshot()
    {

        cameraObject.targetTexture = renderTexture;
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = renderTexture;
        cameraObject.Render();
        mainCameraTexture = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);
        mainCameraTexture.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        mainCameraTexture.Apply();
        RenderTexture.active = currentRT;
        // Get the raw byte info from the screenshot
        byte[] imageBytes = mainCameraTexture.GetRawTextureData();
        cameraObject.targetTexture = null;
        return imageBytes;
    }

    void OnRenderImage(RenderTexture src, RenderTexture dest)
    {
        Vector4 offsets = new Vector4(UnityEngine.Random.value,UnityEngine.Random.value,UnityEngine.Random.value,0);
        noiseMaterial.SetVector("_offsets", offsets);
        if(dest != null)
        {
            Graphics.Blit(src, dest, noiseMaterial);
        }
        else
        {
            RenderTexture temp = RenderTexture.GetTemporary(cameraObject.pixelWidth,cameraObject.pixelHeight);
            Graphics.Blit(src, temp, noiseMaterial);
            Graphics.Blit(temp, dest, new Vector2(1.0f, -1.0f), new Vector2(0.0f, 1.0f));
            RenderTexture.ReleaseTemporary(temp);
        }

    }

    void CaptureAndSendImage()
    {
        isCapturing = true;
        m_TimeLastImageSeconds = Clock.Now;
        m_TimeNextImageSeconds = m_TimeLastImageSeconds + PublishPeriodSeconds;


        byte[] rawImageData = CaptureScreenshot();

        uint imageHeight = (uint)renderTexture.height;
        uint imageWidth = (uint)renderTexture.width;

        var timestamp = new TimeStamp(Clock.time);
        var rosImage = new ImageMsg {
            header = new HeaderMsg
            {
                frame_id = FrameId,
                stamp = new TimeMsg
                {
                    sec = (uint)timestamp.Seconds,
                    nanosec = timestamp.NanoSeconds,
                }
            },
            width = imageWidth,
            height = imageHeight,
            encoding = "rgb8",
            is_bigendian = isBigEndian,
            step = step,
            data = rawImageData
        };

        m_Ros.Publish(topicImage, rosImage);

        isCapturing = false;
        var now = (float)Clock.time;
        if (now > m_TimeNextImageSeconds)
        {
            Debug.LogWarning($"Failed to capture image started at {m_TimeLastImageSeconds:F} before next image was " +
                             $"scheduled to start: {m_TimeNextImageSeconds:F}, rescheduling to now ({now:F})");
            m_TimeNextImageSeconds = now;
        }

        Destroy(mainCameraTexture);
    }

    public void Update()
    {

        if (!isCapturing)
        {
            if (Clock.NowTimeInSeconds + PublishPeriodSeconds <= m_TimeNextImageSeconds)
            {
                return;
            }
            // if(numImages % 100 == 0)
            // {
            //     // Send Camera Info
            //     sendCameraInfo();
            // }
            CaptureAndSendImage();
            sendCameraInfo();
            numImages++;
        }
    }
}
