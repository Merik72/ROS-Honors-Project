using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public class MakeSpawns : MonoBehaviour
{
    [SerializeField]
    private float magnitude_x;
    [SerializeField]

    private float magnitude_y;
    [SerializeField] 
    private float spawn_number;
    //Quaternion spawn_rotation;
    private void Awake()
    {
        spawn_number = 51 / 3;
        for(int i = 0; i < spawn_number; i++)
        {
            GameObject spawnLocation = new GameObject();
            float randomAngle = Random.Range(-180f, 180f);
            spawnLocation.transform.Rotate(Vector3.up, randomAngle);
            spawnLocation.transform.parent = transform;
            float randomX = Random.Range(-0.5f, 0.5f);
            float randomZ = Random.Range(-0.5f, 0.5f);
            spawnLocation.transform.localPosition = new Vector3(randomX, 0, randomZ);
        }
    }
}
