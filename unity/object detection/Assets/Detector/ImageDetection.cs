using System;
using System.Collections;
using System.Collections.Generic;
using TensorFlow;
using UnityEngine;
using UnityEngine.UI;


public class ImageDetection : MonoBehaviour {
    int inputSize = 320;

    public RawImage screen;
    public RectController rectController;
    
    public Texture2D testImage;
    TFSession.Runner runner;
    void Start () {
        TextAsset graphModel = Resources.Load ("freezed_coco_yolo") as TextAsset;
        var graph = new TFGraph ();
        graph.Import (new TFBuffer (graphModel.bytes));
        var session = new TFSession (graph);
        Debug.Log ("loaded freezed graph");

        
        Texture2D input_image = Texture_tool.PaddingScaled (testImage, inputSize, inputSize);
        TFTensor input_tensor = Texture_tool.TransformInput (input_image.GetPixels32 (), inputSize, inputSize);
        SetScreen (testImage.width, testImage.height, screen, testImage);
        
        

        this.runner = session.GetRunner ();
        this.runner.AddInput (graph["input_image"][0], input_tensor);
        this.runner.Fetch (
            graph["output_num"][0],
            graph["output_scores"][0],
            graph["output_classes"][0],
            graph["output_boxes"][0]
        );
        RunGraph(testImage);
        

    }
    void RunGraph(Texture2D image){
        TFTensor[] result = this.runner.Run ();
        int out_num = ((int[]) result[0].GetValue ()) [0];

        if (out_num > 0) {
            var out_scores = (float[]) result[1].GetValue ();
            var out_classes = (int[]) result[2].GetValue ();
            var out_boxes = (float[, ]) result[3].GetValue ();
            rectController.SetRects(image.width, image.height,out_boxes,out_scores,out_classes,this.inputSize,this.inputSize);
        }
    }


    
    void SetScreen (float width, float height, RawImage image, Texture2D input) {
        image.texture = input;
        image.rectTransform.sizeDelta = new Vector2 (width, height);
    }

    // Update is called once per frame
    void Update () {

    }

    
}