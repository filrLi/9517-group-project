import cv2
import numpy as np
import os
from sort.sort import *


window_name = "YOLOv3 + SORT"
confidence_threshold = 0.5
nms_threshold = 0.4
image_size = 416
use_yolov3_tiny = False

image_sequence_path = "./task2/data/sequence"
labels_path = "./task2/yolov3/coco.names"

if use_yolov3_tiny:
    output_video_path = "./task2/video_yolov3_tiny.avi"
    yolo_config_path = "./task2/yolov3/yolov3-tiny.cfg"
    yolo_weights_path = "./task2/yolov3/yolov3-tiny.weights"
else:
    output_video_path = "./task2/video_yolov3.avi"
    yolo_config_path = "./task2/yolov3/yolov3.cfg"
    yolo_weights_path = "./task2/yolov3/yolov3.weights"

labels = []
with open(labels_path, 'rt') as f:
    labels = f.read().rstrip('\n').split('\n')

np.random.seed(42)
label_colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


class Bbox:
    def __init__(self, box, id, age=0):
        self.box = box
        self.id = int(id)
        self.age = age

    def add_age(self):
        self.age += 1


def detect_person(image):
    height, width, _ = image.shape

    blob = cv2.dnn.blobFromImage(
        image, 1/255.0, (image_size, image_size), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(layer_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if labels[class_id] != "person":
                continue

            confidence = scores[class_id]
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([width, height, width, height])
                center_x, center_y, box_width, box_height = box.astype("int")
                if use_yolov3_tiny:
                    x = int(center_x - box_width)
                    y = int(center_y - box_height)
                    boxes.append(
                        [x, y, int(box_width * 2), int(box_height * 2)])
                else:
                    x = int(center_x - box_width / 2)
                    y = int(center_y - box_height / 2)
                    boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids


def draw_detections(image, indexes, boxes):
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y = (boxes[i][0], boxes[i][1])
            w, h = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in label_colors[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            #text = f"{labels[class_ids[i]]}: {confidences[i]:.4f}"
            text = ""
            cv2.putText(image, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


x1, y1, x2, y2 = None, None, None, None
drawing = False
hasDrawn = False
existing_object_ids = []
detected_object_ids = []
num_of_moving_in = 0
num_of_moving_out = 0


def is_overlap(box1, box2):
    box1_x1 = min(box1[0], box1[2])
    box1_y1 = min(box1[1], box1[3])
    box1_x2 = max(box1[0], box1[2])
    box1_y2 = max(box1[1], box1[3])

    box2_x1 = min(box2[0], box2[2])
    box2_y1 = min(box2[1], box2[3])
    box2_x2 = max(box2[0], box2[2])
    box2_y2 = max(box2[1], box2[3])

    if box1_x1 >= box2_x2 or box1_x2 <= box2_x1:
        return False

    if box1_y1 >= box2_y2 or box1_y2 <= box2_y1:
        return False

    return True


def is_contained(box1, box2):
    box1_x1 = min(box1[0], box1[2])
    box1_y1 = min(box1[1], box1[3])
    box1_x2 = max(box1[0], box1[2])
    box1_y2 = max(box1[1], box1[3])

    box2_x1 = min(box2[0], box2[2])
    box2_y1 = min(box2[1], box2[3])
    box2_x2 = max(box2[0], box2[2])
    box2_y2 = max(box2[1], box2[3])

    return box1_x1 >= box2_x1 and\
        box1_x2 <= box2_x2 and\
        box1_y1 >= box2_y1 and\
        box1_y2 <= box2_y2


def draw_rectangle(event, x, y, flags, param):
    global x1, y1, x2, y2, drawing, hasDrawn
    origin_image = param[0]
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y
        cv2.imshow(window_name, origin_image)
        hasDrawn = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.imshow(window_name, origin_image)
            image = origin_image.copy()
            cv2.rectangle(image, (x1, y1), (x, y), (0, 255, 0), 1)
            cv2.imshow(window_name, image)
            x2, y2 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        image = origin_image.copy()
        cv2.rectangle(image, (x1, y1), (x, y), (0, 255, 0), 1)
        cv2.imshow(window_name, image)
        drawing = False
        x2, y2 = x, y
        hasDrawn = True


cap = cv2.VideoCapture(f"{image_sequence_path}/%06d.jpg", cv2.CAP_IMAGES)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, 10.0,
                      (frame_width * 2, frame_height * 2))

#trackers = cv2.MultiTracker_create()
mot_tracker = Sort()
prev = {}
colors = {}

frame_index = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    if frame_index == 0:
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, draw_rectangle, [frame])
        while True:
            if drawing == False and hasDrawn == False:
                cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                break
    else:
        cv2.setMouseCallback(window_name, lambda *args: None)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    if hasDrawn:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    boxes, confidences, class_ids = detect_person(frame)

    indexes = cv2.dnn.NMSBoxes(
        boxes, confidences, confidence_threshold, nms_threshold)

    #draw_detections(frame, indexes, boxes)
    detections = []
    for i in indexes.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        detections.append([x, y, x + w, y + h])

    current = {}
    for object_id, bbox in prev.items():
        overlapping = False
        for det in detections:
            if is_overlap(det, bbox.box) == True:
                overlapping = True
                break
        if overlapping == False and bbox.age < 5:
            bbox.add_age()
            current[object_id] = bbox
            detections.append(bbox.box)

    trackers = mot_tracker.update(np.array(detections))

    for det in trackers:
        x0, y0, x, y, object_id = det[0], det[1], det[2], det[3], int(det[4])
        if object_id not in current:
            bbox = Bbox([x0, y0, x, y], object_id)
            current[object_id] = bbox

    for object_id, bbox in current.items():
        x0, y0, x, y = bbox.box
        x0, y0, x, y = int(x0), int(y0), int(x), int(y)
        if object_id in colors:
            color = colors[object_id]
        else:
            color = tuple(np.random.randint(0, 255) for _ in range(3))
            colors[object_id] = color

        cv2.rectangle(frame, (x0, y0), (x, y), color, 2)
        text = f"{object_id}"
        cv2.putText(frame, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if hasDrawn:
            box1 = (x0, y0, x, y)
            box2 = (x1, y1, x2, y2)
            if is_overlap(box1, box2):
                if object_id not in detected_object_ids and object_id not in existing_object_ids:
                    existing_object_ids.append(object_id)
                    if not is_contained(box1, box2):
                        num_of_moving_in += 1 if frame_index > 0 else 0
            else:
                if object_id in existing_object_ids and object_id not in detected_object_ids:
                    existing_object_ids.remove(object_id)
                    detected_object_ids.append(object_id)
                    num_of_moving_out += 1
            cv2.putText(frame, "No. of moving in: {}".format(num_of_moving_in),
                        (20, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "No. of moving out: {}".format(num_of_moving_out),
                        (20, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(2) & 0xFF
    if frame_index > 0 and key == ord("q"):
        break

    resized_frame = cv2.resize(frame, (frame_width * 2, frame_height * 2))
    out.write(resized_frame)

    frame_index += 1
    prev = current

cap.release()
out.release()
cv2.destroyAllWindows()
