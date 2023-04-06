# Inference-tflite
This repository contains instructions and scripts to infer the Yolo-fastest Model converted to tflite. When you insert image data with input, the inferred label is saved as Yolo-Format. You can also test tflite quantized to '--quant'.

    usage: Run TF-Lite YOLO-V3 Tiny inference. [-h] --input INPUT --output OUTPUT --model MODEL 
                                           [--quant] [--shape SHAPE] [--classes CLASSES] 
                                           [--anchors ANCHORS][-t THRESHOLD] 

    Run TF-Lite YOLO-fastest inference: error: the following arguments are required:
    --input/-i, --output/-o, --model/-m
    
    optional arguments:
      -h, --help              show this help message and exit
      --model MODEL           Model to load.
      --anchors ANCHORS       Anchors file.
      --classes CLASSES       Classes (.names) file.
      --threshold THRESHOLD   Detection threshold. default 0.25
      --quant QUANT           Indicates whether the model is quantized.
      --input Images          Run inference on image.
      --output Output         Save inference results in yolo format
      --shape Shape           Model Input Size (ex. 416x416)
