import cv2

PATH = './data.nosync/sequence'
cap = cv2.VideoCapture(PATH + "/%06d.jpg", cv2.CAP_IMAGES)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

out = cv2.VideoWriter("./task1/output.avi", fourcc,
                      10.0, (frame_width*2, frame_height*2))

ret, frame1 = cap.read()

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while cap.isOpened():
    ret, frame2 = cap.read()
    if frame2 is None:
        break
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    dilated, contours, _ = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 900:
            continue
        count += 1
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame1, "#People: {}".format(count), (20, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 3)
    # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    image = cv2.resize(frame1, (frame_width*2, frame_height*2))
    out.write(image)
    # cv2.imshow("feed", frame1)
    frame1 = frame2

    if cv2.waitKey(40) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()