import cv2
import numpy as np
import os 

class CameraService:
    
    def Load_Yolo_model(self):
        
        folderpath = "Object_Tracking_Core/Services/Yolo_Files/"
        weights_path = os.path.join(folderpath, "yolov4.weights")
        cfg_path = os.path.join(folderpath, "yolov4.cfg")
        classes_path = os.path.join(folderpath, "coco.names")

        net = cv2.dnn.readNet(weights_path, cfg_path)
        with open (classes_path, "r") as f:
            classes= [line.strip() for line in f.readlines()]
        return net,classes
    
    def prepare_frame(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        return blob
    
    def process_predictions(self, outs, frame, target_classes=None):
        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5 and (target_classes is None or class_id in target_classes):
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Wende Non-Maximum Suppression (NMS) an
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Initialisiere leere Listen für die bereinigten Ergebnisse
        final_boxes = []
        final_confidences = []
        final_class_ids = []

        if len(indices) > 0:  # Überprüfen, ob NMS gültige Boxen zurückgibt
            for i in indices.flatten():  # `.flatten()` für einfache Indizes
                final_boxes.append(boxes[i])
                final_confidences.append(confidences[i])
                final_class_ids.append(class_ids[i])

        return final_boxes, final_confidences, final_class_ids
    
    def bounding_boxes(self, frame, boxes, confidences, class_ids, classes):
        for i in range(len(boxes)):
            x ,y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w , y +h), (0,255, 0), 2)
            cv2.putText(frame, f"{label} {confidences[i]:.2f}" ,(x, y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def stop_tracking(self, cap):
        cap.release()
        cv2.destroyAllWindows()