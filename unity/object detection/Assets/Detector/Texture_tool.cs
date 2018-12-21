using System;
using System.Collections;
using System.Collections.Generic;
using Newtonsoft.Json;
using TensorFlow;
using UnityEngine;
using UnityEngine.UI;

[Serializable]
public class LabelsMap {
    public int id { get; set; }
    public string name { get; set; }
}

public static class Texture_tool  {
   

    public static TFTensor TransformInput (Color32[] pic, int width, int height) {
        System.Array.Reverse(pic);
        byte[] floatValues = new byte[width * height * 3];

        for (int i = 0; i < pic.Length; ++i) {

            var color = pic[i];

            floatValues[i * 3 + 0] = color.r;
            floatValues[i * 3 + 1] = color.g;
            floatValues[i * 3 + 2] = color.b;

        }

        TFShape shape = new TFShape ( height, width,3);

        return TFTensor.FromBuffer (shape, floatValues, 0, floatValues.Length);
    }
    public static int[] PaddingBoxScaled (float xmin,float ymin,float xmax,float ymax, int width,int height,int to_width,int to_height) {
        var length=Mathf.Max(to_width,to_height);
       
        var scale= Mathf.Max((float)width/(float)length,(float)height/(float)length);
        var h_offset=(length-to_width)/2;
        var w_offset=(length-to_height)/2;

        int[] result=new int[4]{ 
            Mathf.RoundToInt(xmin*width/scale-w_offset),
            Mathf.RoundToInt(ymin*height/scale-h_offset),
            Mathf.RoundToInt((xmax-xmin)*width/scale),
            Mathf.RoundToInt((ymax-ymin)*height/scale)  };

        return result;
    }
    public static int[] BoxScaled (float xmin,float ymin,float xmax,float ymax, int width,int height) {
        int[] result=new int[4]{ 
            Mathf.RoundToInt(xmin*width),
            Mathf.RoundToInt(ymin*height),
            Mathf.RoundToInt((xmax-xmin)*width),
            Mathf.RoundToInt((ymax-ymin)*height)  };

        return result;
    }
    public static Texture2D Scaled (Texture2D tex, int width, int height, FilterMode mode = FilterMode.Trilinear) {
        Rect texR = new Rect (0, 0, width, height);
        //set Mipmaps
        _gpu_scale (tex, width, height, mode);

        // Update new texture
        Texture2D result = new Texture2D (width, height, TextureFormat.ARGB32, true);
        //result.Resize (width, height);
        result.ReadPixels (texR, 0, 0, true);
        result.Apply (true);
        return result;
    }
   
    public static Texture2D PaddingScaled (Texture2D tex, int width, int height, FilterMode mode = FilterMode.Trilinear) {
        float length = (float) Mathf.Max (tex.width, tex.height);
        float scale = Mathf.Min (width / length, height / length);
        int iw = Mathf.RoundToInt (scale * tex.width);
        int ih = Mathf.RoundToInt (scale * tex.height);
        Rect texR = new Rect (0, 0, iw, ih);
        _gpu_scale (tex, iw, ih, mode);
        int tw = Mathf.Abs (iw - width) / 2;
        int th = Mathf.Abs (ih - height) / 2;
        // Update new texture
        Texture2D result = new Texture2D (width, height, TextureFormat.RGB24, true);
        //result.Resize(width, height);
        result.ReadPixels (texR, tw, th, true);
        result.Apply (true);
        return result;
    }
    public static Texture2D Crop (WebCamTexture tex) {
        

        float length = (float) Mathf.Max (tex.width, tex.height);
        float shorth = (float) Mathf.Min (tex.width, tex.height);
        int ch = Mathf.RoundToInt ((length- tex.width) /2);
        int cw = Mathf.RoundToInt ((length- tex.height)/2);
        Texture2D webcamTexture2D=new Texture2D ( (int)shorth, (int)shorth, TextureFormat.ARGB32,false);
        webcamTexture2D.SetPixels(tex.GetPixels(cw,ch,(int)shorth,(int)shorth));
        webcamTexture2D.Apply ();
        return webcamTexture2D;
    }
    public static void _gpu_scale (Texture2D src, int width, int height, FilterMode fmode) {
        //We need the source texture in VRAM because we render with it
        src.filterMode = fmode;
        src.Apply (true);

        //Using RTT for best quality and performance. Thanks, Unity 5
        RenderTexture rtt = new RenderTexture (width, height, 32);

        //Set the RTT in order to render to it
        Graphics.SetRenderTarget (rtt);

        //Setup 2D matrix in range 0..1, so nobody needs to care about sized
        GL.LoadPixelMatrix (0, 1, 1, 0);

        //Then clear & draw the texture to fill the entire RTT.
        GL.Clear (true, true, new Color (0, 0, 0, 0));
        Graphics.DrawTexture (new Rect (0, 0, 1, 1), src);
    }
}