import sys

import numpy as np
import cv2
import math
import os
import tensorflow as tf
import heapq
import argparse
from tqdm import tqdm
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

OUTPUT_WIDTH = [26, 13]
NUM_BOXES_PER_BLOCK = 3
INPUT_SIZE = 416

# object_names
labels = ["Pothole",
          "Fatigue-Crack",
          "Vertical-Crack",
          "Horizontal-Crack",
          "PoorFixed-Road",
          "Trash",
          "Banner",
          "RoadMark-Poor",
          "SafetyRod-Poor",
          "Manhole"]

ANCHORS = [115,73, 119,199, 242,238, 12,18, 37,49, 52,132]
MASKS = [[3, 4, 5], [0, 1, 2]]
THRESHOLD = 0.25
mNmsThresh = 0.01
ORG_WIDTH = 1920
ORG_HEIGHT = 1080
MODEL_WIDTH = 416
MODEL_HEIGHT = 416

def get_obj_names(names_file):
    labels=[]
    lines=[]
    with open(names_file, 'r') as f:
        lines = [i.strip() for i in f.readlines()]
    for i in lines:
        labels.append(i)
    return labels

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def change_ratio(box):
    x1_model_size = box[0]
    y1_model_size = box[1]
    x2_model_size = box[2]
    y2_model_size = box[3]
    x1_img_size = (ORG_WIDTH/MODEL_WIDTH)*x1_model_size
    y1_img_size = (ORG_HEIGHT/MODEL_HEIGHT)*y1_model_size
    x2_img_size = (ORG_WIDTH/MODEL_WIDTH)*x2_model_size
    y2_img_size = (ORG_HEIGHT/MODEL_HEIGHT)*y2_model_size

    w = (x2_img_size - x1_img_size)/ORG_WIDTH
    h = (y2_img_size - y1_img_size)/ORG_HEIGHT
    x = (x1_img_size + (x2_img_size - x1_img_size)/2)/ORG_WIDTH
    y = (y1_img_size + (y2_img_size - y1_img_size)/2)/ORG_HEIGHT
    return (x,y,w,h)

def nms(result_list):
    nms_list = []
    for k in range(len(labels)):
        pq = [] # 최소힙
        for i in range(len(result_list)):
            if result_list[i][-1] == k:
                heapq.heappush(pq, (-result_list[i][2], result_list[i]))
        while len(pq) > 0:
            _, max = heapq.heappop(pq)
            nms_list.append(max)
            detections = [d for _, d in pq]
            pq.clear()
            for j in range(1, len(detections)):
                detection = detections[j]
                b = detection[3]
                if box_iou(max[3], b) < mNmsThresh:
                    heapq.heappush(pq, (-detection[2], detection))
    return nms_list

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)

def box_intersection(a, b):
    a_left = a[0]; a_top = a[1]; a_right = a[2]; a_bottom = a[3]
    b_left = b[0]; b_top = b[1]; b_right = b[2]; b_bottom = b[3]
    w = overlap((a_left + a_right) / 2, a_right - a_left, (b_left + b_right) / 2, b_right - b_left)
    h = overlap((a_top + a_bottom) / 2, a_bottom - a_top, (b_top + b_bottom) / 2, b_bottom - b_top)
    if w < 0 or h < 0:
        return 0
    area = w * h
    return area

def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2
    l2 = x2 - w2 / 2
    left = l1 if l1 > l2 else l2
    r1 = x1 + w1 / 2
    r2 = x2 + w2 / 2
    right = r1 if r1 < r2 else r2
    return right - left

def box_union(a, b):
    a_left = a[0]; a_top = a[1]; a_right = a[2]; a_bottom = a[3]
    b_left = b[0]; b_top = b[1]; b_right = b[2]; b_bottom = b[3]
    i = box_intersection(a, b)
    u = (a_right - a_left) * (a_bottom - a_top) + (b_right - b_left) * (b_bottom - b_top) - i
    return u

def get_detections_for_keras_float32(interpreter, img_path, shape):

    org_img = cv2.imread(img_path)
    bgr_img = cv2.resize(org_img, shape)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB

    # Normalize image from 0 to 1
    img = np.divide(rgb_img, 255.0).astype(np.float32)
    # Add dimensions
    img = np.expand_dims(img, 0) # (416, 416, 3) -> (1, 416, 416, 3)

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output_width_length = len(OUTPUT_WIDTH)

    detections = []
    for i in range(output_width_length):
        grid_width = OUTPUT_WIDTH[i]
        out = interpreter.get_tensor(output_details[i]['index'])
        for y in range(grid_width):
            for x in range(grid_width):
                for b in range(NUM_BOXES_PER_BLOCK):
                    offset = (grid_width * (NUM_BOXES_PER_BLOCK * (len(labels) + 5))) * y + (
                                NUM_BOXES_PER_BLOCK * (len(labels) + 5)) * x + (len(labels) + 5) * b
                    confidence = 1 / (1 + math.exp(-out[0, y, x, 4])) # sigmoid
                    detected_class = -1
                    max_class = 0
                    classes = out[0, y, x, 5:(5 + len(labels))]
                    classes = softmax(classes)
                    for c in range(len(labels)):
                        if classes[c] > max_class:
                            detected_class = c
                            max_class = classes[c]
                    confidence_in_class = max_class * confidence
                    if confidence_in_class > 0.25:
                        print(img_path)
                        print(detected_class)
                    if confidence_in_class > THRESHOLD:
                        x_pos = (x + (1 / (1 + math.exp(-out[0, y, x, 0])))) * (INPUT_SIZE / grid_width) # sigmoid
                        y_pos = (y + (1 / (1 + math.exp(-out[0, y, x, 1])))) * (INPUT_SIZE / grid_width) # sigmoid
                        w = math.exp(out[0, y, x, 2]) * ANCHORS[2 * MASKS[i][b] + 0]
                        h = math.exp(out[0, y, x, 3]) * ANCHORS[2 * MASKS[i][b] + 1]
                        rect = [max(0, x_pos - w / 2), max(0, y_pos - h / 2), min(416 - 1, x_pos + w / 2), min(416 - 1, y_pos + h / 2)]
                        detections.append(
                            [str(offset), labels[detected_class], confidence_in_class, rect, detected_class])
    return detections


def get_detections_for_keras_int8(interpreter, img_path, shape):
    org_img = cv2.imread(img_path)
    bgr_img = cv2.resize(org_img, shape)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB

    # Add dimensions
    img = np.expand_dims(rgb_img, 0) # (416, 416, 3) -> (1, 416, 416, 3)

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output_width_length = len(OUTPUT_WIDTH)

    detections = []
    for i in range(output_width_length):
        grid_width = OUTPUT_WIDTH[i]
        out = interpreter.get_tensor(output_details[i]['index'])
        o_scale, o_zero = output_details[i]['quantization']
        out_f = (out.astype(np.float32) - o_zero) * o_scale
        for y in range(grid_width):
            for x in range(grid_width):
                for b in range(NUM_BOXES_PER_BLOCK):
                    offset = (grid_width * (NUM_BOXES_PER_BLOCK * (len(labels) + 5))) * y + (
                                NUM_BOXES_PER_BLOCK * (len(labels) + 5)) * x + (len(labels) + 5) * b
                    confidence = 1 / (1 + math.exp(-out_f[0, y, x, 4])) # sigmoid
                    detected_class = -1
                    max_class = 0
                    classes = out_f[0, y, x, 5:(5 + len(labels))]                 
                    classes = softmax(classes)
                    for c in range(len(labels)):
                        if classes[c] > max_class:
                            detected_class = c
                            max_class = classes[c]
                    confidence_in_class = max_class * confidence
                    if confidence_in_class > THRESHOLD:
                        x_pos = (x + (1 / (1 + math.exp(-out_f[0, y, x, 0])))) * (INPUT_SIZE / grid_width) # sigmoid
                        y_pos = (y + (1 / (1 + math.exp(-out_f[0, y, x, 1])))) * (INPUT_SIZE / grid_width) # sigmoid
                        w = math.exp(out_f[0, y, x, 2]) * ANCHORS[2 * MASKS[i][b] + 0]
                        h = math.exp(out_f[0, y, x, 3]) * ANCHORS[2 * MASKS[i][b] + 1]
                        rect = [max(0, x_pos - w / 2), max(0, y_pos - h / 2), min(416 - 1, x_pos + w / 2), min(416 - 1, y_pos + h / 2)]
                        detections.append(
                            [str(offset), labels[detected_class], confidence_in_class, rect, detected_class])
    return detections

if __name__=="__main__":
    parser = argparse.ArgumentParser("Run TF-Lite YOLO-fastest inference")
    parser.add_argument("--input", "-i", type=str, required=True, help="Images Path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Save Path Infernece Result")
    parser.add_argument("--model", "-m", type=str, required=True, help="TFLite Model Path")
    parser.add_argument("--quant", "-q", action="store_true", help="Model is Quantized")
    parser.add_argument("--shape", "-s", default="416x416", help="Model Input Size")
    parser.add_argument("--classes", "-c", type=str, help="Object Model Names File Path (ex. -n obj_names.txt")
    parser.add_argument("--anchors", "-a", type=str, help="Anchors File Path (ex. -a best_anchors.txt")
    parser.add_argument("--threshold", "-t", type=float, default=0.25, help="Inference Threshold, default 0.25")
    args = parser.parse_args()

    height, width = args.shape.split('x')
    shape = (int(height), int(width))

    model_path = args.model_path
    output_path = args.path_out
    classes_file = args.classes
    anchors_file = args.anchors
    threshold = args.threshold

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    img_path_list = glob(args.path + "/**/*.jpg", recursive=True) + glob(args.path + "/**/*.png", recursive=True)

    # Load the TF-Lite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()


    for img_path in tqdm(img_path_list):
        fn = os.path.basename(img_path)
        annot_fn = fn.replace('jpg', 'txt')
        annot_fn = annot_fn.replace('png', 'txt')
        annot_path = os.path.join(output_path, annot_fn)
        with open(annot_path, 'w') as f:
            f.write("")

        if args.quant:
            re_list = get_detections_for_keras_int8(interpreter, img_path, shape)
        else:
            re_list = get_detections_for_keras_float32(interpreter, img_path, shape)

        final_list = nms(re_list)
        with open(annot_path, 'w') as f:
            if not final_list:
                f.write("")
            else:
                annot_list = []
                for i in range(len(final_list)):
                    cls, x1, y1, x2, y2 = final_list[i][-1], final_list[i][-2][0], final_list[i][-2][1], final_list[i][-2][2], final_list[i][-2][3]
                    cx, cy, bw, bh = change_ratio((x1, y1, x2, y2))
                    annot_list.append(f"{cls} {cx} {cy} {bw} {bh}\n")
                for i in annot_list:
                    f.write(i)
