using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// TCP stuff
using System;
using System.Threading;
using System.Net;
using System.Net.Sockets;
using System.Text;

public class ModelController : MonoBehaviour
{
    private Animator anim;
    GameObject main_obj;
    public float max_rotation_angle = 45.0f;
    public float mar_max_threshold = 1.0f;
    public float mar_min_threshold = 0.0f;

    private Transform neck;
    private Quaternion neck_quat;


    private float roll = 0, pitch = 0, yaw = 0;
    private float mar = 0;

    // Start is called before the first frame update
    void Start()
    {
        neck_quat = Quaternion.Euler(0, 90, -90);
    }

    // Parse the string received through TCP to
    // Update the parameters
    public void parseMessage(String message) {
        string[] res = message.Split(' ');

        roll = float.Parse(res[0]);
        pitch = float.Parse(res[1]);
        yaw = float.Parse(res[2]);
        mar = float.Parse(res[3]);
    }

    // Update is called once per frame
    void Update()
    {
        HeadRotation();
        MouthMoving();
    }


    void HeadRotation()
    {
        // clamp the angles to prevent unnatural movement
        float pitch_clamp = Mathf.Clamp(pitch, -max_rotation_angle, max_rotation_angle);
        float yaw_clamp = Mathf.Clamp(yaw, -max_rotation_angle, max_rotation_angle);
        float roll_clamp = Mathf.Clamp(roll, -max_rotation_angle, max_rotation_angle);

        // do rotation at neck to control the movement of head
        gameObject.transform.Find("FLAME2020-female/root/neck").localRotation = Quaternion.Euler(-pitch_clamp, yaw_clamp, -roll_clamp);// * neck_quat;
    }
    
    void MouthMoving() {
        float mar_clamped = Mathf.Clamp(mar, mar_min_threshold, mar_max_threshold);
        float ratio = (mar_clamped - mar_min_threshold) / (mar_max_threshold - mar_min_threshold);
        // enlarge it to [0, 100]
        ratio = ratio *  30/ (mar_max_threshold - mar_min_threshold);
        SetMouth(ratio);
    }

    void SetMouth(float ratio)
    {
        gameObject.transform.Find("FLAME2020-female/root/neck/jaw").localRotation = Quaternion.Euler(ratio, 0f, 0f);
    }

}
