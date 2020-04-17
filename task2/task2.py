import cv2
import numpy as np
import os

confidence_threshold = 0.5
nms_threshold = 0.4

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

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
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


def draw_detections(image, indexes):
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y = (boxes[i][0], boxes[i][1])
            w, h = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in label_colors[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = f"{labels[class_ids[i]]}: {confidences[i]:.4f}"
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


cap = cv2.VideoCapture(f"{image_sequence_path}/%06d.jpg", cv2.CAP_IMAGES)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, 10.0, (frame_width * 2, frame_height * 2))

while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (frame_width * 2, frame_height * 2))
    boxes, confidences, class_ids = detect_person(resized_frame)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    draw_detections(resized_frame, indexes)
    out.write(resized_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
