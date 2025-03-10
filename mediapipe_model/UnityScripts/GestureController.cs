using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using TMPro;
using UnityEngine;

public class GestureController : MonoBehaviour
{
    public TextMeshProUGUI text;
    public Light light;
    public Camera camera;
    string pose = "None";
    float timer = 0;
    float bg_timer = 0;
    float metalic_timer = 0;
    float metalic;
    int type = -1;
    float r,
        g,
        b = 0;
    float bg_r,
        bg_g,
        bg_b = 0;

    // Start is called before the first frame update
    void Start() { }

    // Update is called once per frame
    void Update()
    {
        text.text = pose;
        if (pose == "Thumb_Up")
        {
            light.transform.Rotate(new Vector3(0f, 60f * Time.deltaTime, 0f));
        }
        if (pose == "Thumb_Down")
        {
            light.transform.Rotate(new Vector3(0f, -60f * Time.deltaTime, 0f));
        }
        if (pose == "Victory")
        {
            timer += Time.deltaTime;
            r = (0.5f * timer) % 1;
            g = (1f * timer) % 1;
            b = (1.5f * timer) % 1;
            print(string.Format("R: {0:F}; F: {1:F}; B: {2:F}, time: {3:F}", r, g, b, timer));
            gameObject.transform.Find("FLAME-female").GetComponent<Renderer>().material.color =
                new Color(r, g, b);
        }
        if (pose == "Pointing_Up")
        {
            bg_timer += Time.deltaTime;
            bg_r = (0.2f * bg_timer) % 1;
            bg_g = (0.3f * bg_timer) % 1;
            bg_b = (0.4f * bg_timer) % 1;
            print(
                string.Format(
                    "R: {0:F}; F: {1:F}; B: {2:F}, time: {3:F}",
                    bg_r,
                    bg_g,
                    bg_b,
                    bg_timer
                )
            );
            Camera.main.backgroundColor = new Color(bg_r, bg_g, bg_b);
        }
        if (pose == "Open_Palm")
        {
            gameObject.transform.Rotate(new Vector3(0f, -20f * Time.deltaTime, 0f));
        }

        if (pose == "Closed_Fist")
        {
            gameObject.transform.Rotate(new Vector3(0f, 20f * Time.deltaTime, 0f));
        }
    }

    public void parseMessage(String message)
    {
        // string[] res = message.Split(' ');
        type = int.Parse(message.Split('.')[0]);
        setText(type);
    }

    void setText(int type)
    {
        string[] names =
        {
            "Pointing_Up",
            "Open_Palm",
            "Closed_Fist",
            "Victory",
            "ILoveYou",
            "Thumb_Down",
            "Thumb_Up"
        };
        if (type < 0)
        {
            print("None");
            pose = "None";
        }
        else
        {
            print(names[type]);
            pose = names[type];
        }
    }
}
