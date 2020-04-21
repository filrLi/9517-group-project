import cv2
import numpy as np
import os
from sort import *

window_name = "Task1"
confidence_threshold = 0.5
nms_threshold = 0.3
max_age = 5


image_sequence_path = "../data.nosync/sequence"
output_video_path = "./output.avi"


yolo_config_path = "./yolov3/yolov3-tiny.cfg"
yolo_weights_path = "./yolov3/yolov3-tiny.weights"


# setup neural network
net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def detect_person(image):
    height, width, _ = image.shape

    blob = cv2.dnn.blobFromImage(
        image, 1/255.0, (height, width), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(layer_names)

    boxes = []
    confidences = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if class_id != 0:
                continue

            confidence = scores[class_id]
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([width, height, width, height])
                center_x, center_y, box_width, box_height = box.astype("int")
                x = int(center_x - box_width)
                y = int(center_y - box_height)
                boxes.append([x, y, int(2*box_width), int(2*box_height)])
                confidences.append(float(confidence))

    return boxes, confidences


def draw_bbox(frame, bb, id, colors):
    if id in colors:
        color = colors[id]
    else:
        color = tuple(np.random.randint(0, 255) for _ in range(3))
        colors[id] = color

    x, y, x2, y2 = (int(_)for _ in bb)
    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
    text = f"id:{id}"
    cv2.putText(frame, text, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# load files and config
cap = cv2.VideoCapture(f"{image_sequence_path}/%06d.jpg", cv2.CAP_IMAGES)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, 10.0,
                      (frame_width * 2, frame_height * 2))


# set up color map
colors = {}

# Create MultiTracker object
mot_tracker = Sort()
prev = {}

# frame1
ret, frame1 = cap.read()
boxes, confidences = detect_person(frame1)
# apply Non-Maximum Suppression
indexes = cv2.dnn.NMSBoxes(
    boxes, confidences, confidence_threshold, nms_threshold)
detections = [boxes[i] for i in indexes.flatten()]

# convert to [x1,y1,w,h] to [x1,y1,x2,y2]
dets = np.array(detections)
dets[:, 2:4] += dets[:, 0:2]

# define a class to store bbox


class Bbox:
    def __init__(self, box, id, age=0):
        self.box = box
        self.id = id
        self.age = age

    def addAge(self):
        self.age += 1


# get updated location of objects in subsequent frames
track_bbs_ids = mot_tracker.update(dets)
for bb_id in track_bbs_ids:
    bb = bb_id[:-1]
    id = bb_id[-1]
    bbox = Bbox(bb, id)
    prev[id] = bbox


frame_index = 0
while (cap.isOpened()):
    count = 0
    ret, frame = cap.read()
    if not ret:
        break

    print(frame_index)
    frame_index += 1

    boxes, confidences = detect_person(frame)

    # apply Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(
        boxes, confidences, confidence_threshold, nms_threshold)
    detections = [boxes[i] for i in indexes.flatten()]

    # convert [x1,y1,w,h] to [x1,y1,x2,y2]
    dets = np.array(detections)
    dets[:, 2:4] += dets[:, 0:2]

    # compare with prev frame
    current = {}
    for id, bbox in prev.items():
        x1, y1, x2, y2 = bbox.box
        # cannot find box overlap with current prev box
        if dets[(dets[:, 2] > x1) & (dets[:, 3] > y1) & (dets[:, 0] < x2) & (dets[:, 1] < y2)].shape[0] == 0 and bbox.age < max_age:
            bbox.addAge()
            current[id] = bbox
            dets = np.vstack((dets, bbox.box))

    # get updated location of objects in subsequent frames
    track_bbs_ids = mot_tracker.update(dets)

    # draw tracked objects
    for bb_id in track_bbs_ids:
        bb = bb_id[:-1]
        id = bb_id[-1]
        if id not in current:
            bbox = Bbox(bb, id)
            current[id] = bbox
        draw_bbox(frame, bb, id, colors)
        count += 1

    # update prev as current
    prev = current

    resized_frame = cv2.resize(frame, (frame_width * 2, frame_height * 2))
    cv2.putText(frame, "#People: {}".format(count), (20, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 3)
    cv2.imshow(window_name, frame)
    out.write(resized_frame)

    # stop listener
    key = cv2.waitKey(40)
    if key == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
