using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraScript : MonoBehaviour
{

    public GameObject Camera1;
    public GameObject Camera2;
    public GameObject Camera3;
    public GameObject Camera4;


    void Update()
    {
        bool v1 = Input.GetKeyDown("1");
        bool v2 = Input.GetKeyDown("2");
        bool v3 = Input.GetKeyDown("3");

        bool v4 = Input.GetKeyDown(KeyCode.Escape);
       if ( v1 | v2 | v3 | v4)
        CameraChange(v1,v2,v3,v4);
         

        
    }
    public void OnGUI()
    {
        GUIStyle centeredStyle = GUI.skin.GetStyle("Label");
        centeredStyle.alignment = TextAnchor.UpperCenter;
        GUI.Label(new Rect(Screen.width / 2 - 200, 10, 600, 20), "Press 1,2,3, or Escape to select a view.", centeredStyle); 
    }



    void CameraChange(bool v1, bool v2, bool v3, bool v4)
    {
        Camera1.SetActive(v1);
        Camera2.SetActive(v2);
        Camera3.SetActive(v3);
        Camera4.SetActive(v4);
    }
 
}