using System;
using System.Collections;
using System.Collections.Generic;
using Newtonsoft.Json;
using UnityEngine;
using UnityEngine.UI;
[Serializable]
public class Response<T> {
    public List<T> list { get; set; }
}
public class RectController : MonoBehaviour {

    public bool isLableId; //yolo with false for label index , tf zoo with true for label id
    public int maxBoxes = 20;
    List<LabelsMap> classesData;

    public float scoreThreshold = 0.5f;
    public TextAsset classes;
    public Transform tmplGroup;
    public Transform rectTmpl;

    public List<Transform> rectList;
    // Use this for initialization
    void Awake () {
        GenerateRects ();

        this.classesData = JsonConvert.DeserializeObject<List<LabelsMap>> (classes.text);

    }
    void Start () {

    }

    void GenerateRects () {
        for (int i = 0; i < this.maxBoxes; i++) {
            this.rectList.Add (GameObject.Instantiate (this.rectTmpl, this.tmplGroup));
        }
    }
    public void SetRects (int to_width, int to_height, float[, ] boxes, float[] scores, int[] classes, int input_width, int input_height, bool inputPadding = true) {
        for (int i = 0; i < scores.Length && i < this.maxBoxes; i++) {
            if (scores[i] >= this.scoreThreshold) {
                try {
                    int[] boxScaled;
                    if (inputPadding) {
                        boxScaled = Texture_tool.PaddingBoxScaled (boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3], input_width, input_height, to_width, to_height);
                    } else {
                        boxScaled = Texture_tool.BoxScaled (boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3], to_width, to_height);
                    }

                    this.rectList[i].localScale = new Vector2 (1, 1);
                    this.rectList[i].localPosition = new Vector2 (boxScaled[0], boxScaled[1]);
                    this.rectList[i].GetComponent<RectTransform> ().sizeDelta = new Vector2 (boxScaled[2], boxScaled[3]);
                    this.rectList[i].GetChild (0).localScale = new Vector2 (1, 1);
                    if (isLableId) {
                        this.rectList[i].GetChild (0).GetComponent<Text> ().text = String.Format ("{0} {1:0%}", this.classesData.Find (x => x.id == classes[i]).name, scores[i]);
                    } else {
                        this.rectList[i].GetChild (0).GetComponent<Text> ().text = String.Format ("{0} {1:0%}", this.classesData[classes[i]].name, scores[i]);
                    }
                } catch (System.Exception) {
                   SetRectsZero(i);
                }
            }
        }
        for (int i = scores.Length; i >= scores.Length && i < maxBoxes; i++) {
            SetRectsZero(i);
        }
    }
    public void SetRectsZero(int index=-1){
        if (index>-1){
        this.rectList[index].localScale = new Vector2 (0, 0);
        this.rectList[index].GetChild (0).localScale = new Vector2 (0, 0);
        }else{
            for (int i = 0; i <this.maxBoxes; i++) {
            this.rectList[i].localScale = new Vector2 (0, 0);
            this.rectList[i].GetChild (0).localScale = new Vector2 (0, 0);
            }
        }
    }
}