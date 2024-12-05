import cv2
import numpy as np

class Camera_Service:
    def Load_Yolo_model():
        net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        with open ("coco.names", "r") as f:
            classes= [line.strip() for line in f.readlines()]
        return net,classes
    
    def prepare_frame(frame):
        blob = cv2.dnn.blobFromImage(frame, (416, 416), True, crop=False)
        return blob
    
    def process_predictions(outs,frame, traget_classes=None):
        boxes = []
        confidences = []
        class_ids =[]

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and (traget_classes is None or class_id in traget_classes):
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x , y ,w ,h])
                    confidence.append(float(confidence))
                    class_ids.append(class_id)

        return boxes, confidences, class_ids
    
    def bounding_boxes(frame, boxes, confidences, class_ids, classes):
        for i in range(len(boxes)):
            x ,y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w , y +h), (0,255, 0), 2)

    def stop_tracking(cap):
        cap.release()
        cv2.destroyAllWindows()