import cv2
import numpy as np

#Yolo file paths
weights_path = "C:/Users/saipr/OneDrive/Desktop/yoloObjectDetection/yolov3.weights"
config_path = "C:/Users/saipr/OneDrive/Desktop/yoloObjectDetection/yolov3.cfg"
names_path = "C:/Users/saipr/OneDrive/Desktop/yoloObjectDetection/coco.names"

#Loading YOLO
net = cv2.dnn.readNet(weights_path, config_path)
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

#Loading an image
img_path = "C:/Users/saipr/OneDrive/Desktop/yoloObjectDetection/Images/bicycle.jpg"
img = cv2.imread(img_path)
height, width, channels = img.shape

#Detecting the objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

#Information to display
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
color = (0, 255, 0)  # Use a fixed color for debugging
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        print(f"Detected: {label}")
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), font, 1, color, 2)

#Displaying the final image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()