using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;

public class CameraPublish : MonoBehaviour
{
    ROSConnection ros;
    public string topicName = "image_raw";
    public CustomRenderTexture srcRenderTexture;
    public float frameRate = (1 / 90f);
    private float elapsedTime;

    public Camera cam;
    // below code is from https://forum.unity.com/threads/render-camera-image-into-byte-array.793428/
    // private Material mat;
    public int resolutionWidth = 1280;
    public int resolutionHeight = 720;
    public int bytesPerPixel = 3;
    private byte[] rawByteData;
    private Texture2D texture2D;
    private Rect rect;
    private byte[] data;
    RosMessageTypes.Sensor.ImageMsg image;
    public float gaussianNoiseMean = 0;
    public float gaussianNoiseStddev = 0.007f;
    private Material noiseMaterial;
    void Awake()
    {
        noiseMaterial = new Material(Shader.Find("Custom/GaussianNoise"));
        noiseMaterial.SetFloat("_mean", gaussianNoiseMean);
        noiseMaterial.SetFloat("_stddev", gaussianNoiseStddev);
    }
    private void Start()
    {
        //mat = new Material(Shader.Find("Custom/DepthGrayscale"));
        srcRenderTexture = new CustomRenderTexture(resolutionWidth, resolutionHeight);
        cam.targetTexture = srcRenderTexture;
        rawByteData = new byte[resolutionWidth * resolutionHeight * bytesPerPixel];
        texture2D = new Texture2D(resolutionWidth, resolutionHeight, TextureFormat.RGB24, false);
        rect = new Rect(0, 0, resolutionWidth, resolutionHeight);
        // pause borrowed code

        byte[] data = rawByteData;
        image = new RosMessageTypes.Sensor.ImageMsg(
                /*
                header;
                height;
                width;
                encoding;
                is_bigendian;
                step;
                data;
                 */
                new RosMessageTypes.Std.HeaderMsg(),
                 (uint) resolutionWidth,
                 (uint) resolutionHeight,
                "bgr8",
                0,
                0,
                data
            );
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<RosMessageTypes.Sensor.ImageMsg>(topicName);
    }

    // resume borrowed code
    //private void OnRenderImage(RenderTexture source, RenderTexture destination)
    //{
    //    //Graphics.Blit(source, destination /*, mat*/);
    //    RenderTexture.active = renderTexture;
    //    texture2D.ReadPixels(rect, 0, 0);
    //    System.Array.Copy(texture2D.GetRawTextureData(), rawByteData, rawByteData.Length);
    //    byte[] data = rawByteData;

    //    // Order is in https://github.com/Unity-Technologies/ROS-TCP-Connector/blob/main/com.unity.robotics.ros-tcp-connector/Runtime/Messages/Sensor/msg/ImageMsg.cs#L113
    //    RosMessageTypes.Sensor.ImageMsg image = new RosMessageTypes.Sensor.ImageMsg(
    //        /*
    //        header;
    //        height;
    //        width;
    //        encoding;
    //        is_bigendian;
    //        step;
    //        data;
    //         */
    //        new RosMessageTypes.Std.HeaderMsg(),
    //        720,
    //        1280,
    //        "",
    //        0,
    //        0,
    //        data
    //    );
    //    ros.Publish(topicName, image);
    //    Debug.Log("We published an image!");
    //}

    // finish borrowed code
    // Update is called once per frame
    void Update()
    {
        elapsedTime += Time.deltaTime;

        if(elapsedTime >= frameRate)
        {
            //Debug.Log("RENDERED!" + elapsedTime);
            //cam.Render(); 
            //srcRenderTexture.material = noiseMaterial;
            Graphics.Blit(srcRenderTexture, noiseMaterial);
            RenderTexture.active = srcRenderTexture;
            //Graphics.Blit(srcRenderTexture, noiseMaterial);
            texture2D.ReadPixels(rect, 0, 0);
            System.Array.Copy(texture2D.GetRawTextureData(), rawByteData, rawByteData.Length);
            byte[] data = rawByteData;
            image.data = data;
            // Order is in https://github.com/Unity-Technologies/ROS-TCP-Connector/blob/main/com.unity.robotics.ros-tcp-connector/Runtime/Messages/Sensor/msg/ImageMsg.cs#L113
            
            ros.Publish(topicName, image);
            //Debug.Log("We published an image!");
            elapsedTime = 0;
        }
    }
}
