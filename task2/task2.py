import cv2
import numpy as np
import os
from sort.sort import *


window_name = "YOLOv3 + SORT"
confidence_threshold = 0.5
nms_threshold = 0.4
image_size = 416

image_sequence_path = "./task2/data/sequence"
output_video_path = "./task2/video.avi"

labels_path = "./task2/yolov3/coco.names"
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
            text = f"{labels[class_ids[i]]}: {confidences[i]:.4f}"
            cv2.putText(image, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


x1, y1, x2, y2 = None, None, None, None
drawing = False
hasDrawn = False


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

    if hasDrawn:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    frame_index += 1

    boxes, confidences, class_ids = detect_person(frame)

    indexes = cv2.dnn.NMSBoxes(
        boxes, confidences, confidence_threshold, nms_threshold)

    #draw_detections(frame, indexes, boxes)
    detections = []
    for i in indexes.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        detections.append([x, y, x + w, y + h])

    trackers = mot_tracker.update(np.array(detections))
    for d in trackers:
        x0, y0, x, y, object_id = int(d[0]), int(
            d[1]), int(d[2]), int(d[3]), int(d[4])
        color = [int(c) for c in label_colors[object_id]]
        cv2.rectangle(frame, (x0, y0), (x, y), color, 2)
        text = f"{object_id}"
        cv2.putText(frame, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF

    resized_frame = cv2.resize(frame, (frame_width * 2, frame_height * 2))
    out.write(resized_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
