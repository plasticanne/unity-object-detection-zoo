using System;
using System.Collections;
using System.Collections.Generic;
using TensorFlow;
using UnityEngine;
using UnityEngine.UI;


public class WebCamDetection : MonoBehaviour {
    int screenSize = 400;
    int inputSize = 96; //for yolo, require
    int requestedWidth= 360; 
    int requestedHeight= 240;
    int requestedFPS= 30;
    public RawImage screen;
    public RectController rectController;
    public Text info;
    TFGraph graph;
    TFSession session;
    WebCamTexture webcamTexture;

    private float updateInterval = 0.5F;
    private double lastInterval;
    private int frames = 0;
    private float fps;
    void Start () {
        lastInterval = Time.realtimeSinceStartup;
        frames = 0;
        TextAsset graphModel = Resources.Load ("freezed_coco_yolo") as TextAsset;
        this.graph = new TFGraph ();
        graph.Import (new TFBuffer (graphModel.bytes));
        this.session = new TFSession (graph);
        Debug.Log ("loaded freezed graph");

        WebCamDevice[] devices = WebCamTexture.devices;
        this.webcamTexture = new WebCamTexture(devices[0].name, this.requestedWidth, this.requestedHeight,this.requestedFPS);
        this.webcamTexture.Play();
       
        
        
    }
    void RunGraph(Color32[] input_image ){
        var runner = this.session.GetRunner ();
        TFTensor input_tensor = Texture_tool.TransformInput (input_image, inputSize, inputSize);
        runner.AddInput (this.graph["input_image"][0], input_tensor);
        runner.Fetch (
            this.graph["output_num"][0],
            this.graph["output_scores"][0],
            this.graph["output_classes"][0],
            this.graph["output_boxes"][0]
        );
        TFTensor[] result = runner.Run ();
        DrawResult(result);
        
    }
    void DrawResult(TFTensor[] result){
        int out_num = ((int[]) result[0].GetValue (jagged: false)) [0];
        this.info.text=String.Format("fps: {0:N},  results: {1}",this.fps,out_num);
        if (out_num > 0) {
            var out_scores = (float[]) result[1].GetValue (jagged: false);
            var out_classes = (int[]) result[2].GetValue (jagged: false);
            var out_boxes = (float[, ]) result[3].GetValue (jagged: false);
            rectController.SetRects(this.screenSize, this.screenSize,out_boxes,out_scores,out_classes,this.inputSize,this.inputSize,false);
        }else{
            rectController.SetRectsZero(-1);
        }
    }

    void Fps(){
        ++this.frames;
        float timeNow = Time.realtimeSinceStartup;
        if (timeNow > this.lastInterval + this.updateInterval)
        {
            this.fps = (float)(this.frames / (timeNow -this.lastInterval));
            this.frames = 0;
            this.lastInterval = timeNow;
        }
    }
    

    void SetScreen (int width,int height, RawImage image, Texture2D input) {
        image.texture = input;
        image.rectTransform.sizeDelta = new Vector2 (width, height);
    }
    bool isPosing=false;

    // Update is called once per frame
    void Update () {
        Fps();
        Texture2D input_image = Texture_tool.Scaled( Texture_tool.Crop(webcamTexture),this.inputSize, this.inputSize);
        SetScreen ( this.screenSize, this.screenSize,this.screen, input_image);
        if (!this.isPosing) {
            isPosing = true;
            RunGraph(input_image.GetPixels32());
            isPosing = false;
            input_image=null;
        };
    }
    
    
    
}