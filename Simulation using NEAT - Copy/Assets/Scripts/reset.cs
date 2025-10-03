using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class reset : MonoBehaviour
{
    public List<GameObject> Dingos = new List<GameObject>();
    private void Start()
    {
        foreach(GameObject go in Dingos)
       Dingos.Add(gameObject);
    }
}
