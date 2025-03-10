// Socket
using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;

public class SocketServer : MonoBehaviour
{
    // the controller script attached to the live2d / 3D model
    ModelController modelController;
    GestureController gestureController;
    public SkinnedMeshRenderer skinnedMeshRenderer;

    Thread receiveThread;

    // the client connected to the TCP server
    TcpClient client;

    // Unity side
    TcpListener server;

    bool serverUp = false;

    string message = "";
    bool flameFlag = true;

    [SerializeField]
    int port = 5066;
    int portMin = 5000;
    int portMax = 6000;

    // Start is called before the first frame update
    void Start()
    {
        modelController = GameObject.Find("FLAME").GetComponent<ModelController>();
        gestureController = GameObject.Find("FLAME").GetComponent<GestureController>();
        InitTCP();
    }

    // Update is called once per frame
    void Update()
    {
        // 
        if (message != "" && flameFlag)
        {
            parseFlame(message);
            flameFlag = false;
        }
    }

    // Init the TCP server
    // Attach this to the OnClick listener of Start button on TCP UI panel
    public void InitTCP()
    {
        try
        {
            // local host
            server = new TcpListener(IPAddress.Parse("127.0.0.1"), port);
            server.Start();

            serverUp = true;

            // create a thread to accept client
            receiveThread = new Thread(new ThreadStart(ReceiveData));
            receiveThread.IsBackground = true;
            receiveThread.Start();
        }
        catch (Exception e)
        {
            // usually error occurs if the port is used by other program.
            // a "SocketException: Address already in use" error will show up here
            print(e.ToString());
        }
    }

    // Stop the TCP server
    // Attach this to the OnClick listener of Stop button on TCP UI panel
    public void StopTCP()
    {
        if (!serverUp)
            return;

        if (client != null)
            client.Close();

        server.Stop();

        print("Server is off.");

        if (receiveThread.IsAlive)
            receiveThread.Abort();

        serverUp = false;
    }

    private void ReceiveData()
    {
        try
        {
            // Buffer
            Byte[] bytes = new Byte[4096];

            while (true)
            {
                print("Waiting for a connection...");

                client = server.AcceptTcpClient();
                print("Connected!");

                // I/O Stream for sending/ receiving to/ from client
                NetworkStream stream = client.GetStream();

                int length;

                while ((length = stream.Read(bytes, 0, bytes.Length)) != 0)
                {
                    var incommingData = new byte[length];
                    Array.Copy(bytes, 0, incommingData, 0, length);
                    string clientMessage = Encoding.ASCII.GetString(incommingData);
                    // print(clientMessage);
                    // call model controller to update values
                    if (clientMessage.StartsWith("9.9999"))
                    {
                        modelController.parseMessage(clientMessage.Substring(7));
                    }
                    if (clientMessage.StartsWith("8.8888"))
                    {
                        gestureController.parseMessage(clientMessage.Substring(7));
                        // print(clientMessage);
                    }
                    if (clientMessage.StartsWith("1.1111"))
                    {
                        message = clientMessage.Substring(7);
                        // print(clientMessage);
                    }
                }
            }
        }
        catch (Exception e)
        {
            print(e.ToString());
        }
    }

    public void parseFlame(String message)
    {
        StartCoroutine(changeFace(message));
    }

    IEnumerator changeFace(String message)
    {
        string[] res = message.Split(' ');
        float shapeVal;
        for (int i = 0; i < res.Length; i++)
        {
            if (float.TryParse(res[i], out shapeVal))
            {
                skinnedMeshRenderer.SetBlendShapeWeight(
                    i,
                    // k,
                    shapeVal * 100
                );
            }
        }
        yield break;    
    }
    public void ConfirmPort()
    {
        int temp;
        bool success = Int32.TryParse("5066", out temp);
        port = temp;
    }

    void OnApplicationQuit()
    {
        // close the TCP stuffs when the application quits
        StopTCP();
    }
}
