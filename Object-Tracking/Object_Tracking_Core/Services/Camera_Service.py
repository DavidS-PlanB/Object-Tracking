import cv2
import numpy as np
import os 
from typing import Optional
import dotenv

class CameraService:

    _instance = None
    
    def __init__(self):
        dotenv.load_dotenv()
        '''Lädt das YOLO-Modell sowie die zugehörigen Klassen.'''
        folderpath = os.getenv("FolderPath")
        weights_path = os.path.join(folderpath, os.getenv("Weights"))
        cfg_path = os.path.join(folderpath, os.getenv("Cfg"))
        classes_path = os.path.join(folderpath, os.getenv("Coco"))

        #print("Folder path:", folderpath)
        #print("weights path: ", weights_path)
        #print("config path: ",cfg_path)
        #print("Classes path: ", classes_path)

        self.net = cv2.dnn.readNet(weights_path, cfg_path)

        with open (classes_path, "r") as f:
            self.classes= [line.strip() for line in f.readlines()]

    def Load_Yolo_model(self) :
        return self.net, self.classes
    
    def prepare_frame(self, frame: np.ndarray):
        '''Wandelt ein Bild in ein Blob um, das als Eingabe für das YOLO-Modell dient.'''
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        return blob
    
    def process_predictions(self, outs: list[np.ndarray], frame: np.ndarray, target_classes : Optional[list[int]] = None):
        ''' Verarbeitet die Ausgabe des YOLO-Modells, um Bounding-Boxen, Konfidenzwerte und Klassen-IDs zu extrahieren.'''
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
    
    def bounding_boxes(self, frame :np.ndarray, boxes: list[list[int]], confidences: list[float], class_ids: list[int], classes: list[str]):
        ''' Zeichnet Bounding-Boxen, Labels und Konfidenzwerte auf ein Bild.'''
        for i in range(len(boxes)):
            x ,y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w , y +h), (0,255, 0), 2)
            cv2.putText(frame, f"{label} {confidences[i]:.2f}" ,(x, y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def stop_tracking(self, cap: cv2.VideoCapture):
        '''Beendet die Videoaufnahme und schließt alle Fenster.'''
        cap.release()
        cv2.destroyAllWindows()

    def complet_detection(self, frame, target_classes):
        blob = self.prepare_frame(frame)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        boxes, confidences, class_ids = self.process_predictions(outs, frame, target_classes)
        self.bounding_boxes(frame, boxes, confidences, class_ids, self.classes)
        cv2.imshow("Detected Objects", frame)