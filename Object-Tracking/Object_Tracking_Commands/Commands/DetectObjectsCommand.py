import cv2
from Object_Tracking_Commands.Commands.Command import Command
from Object_Tracking_Core.Services.Camera_Service import CameraService

class DetectObjectsCommand(Command):
    def __init__(self, cameraService: CameraService, frame, cap, target_classes=None):
        self.service = cameraService
        self.frame = frame
        self.cap = cap
        self.target_classes = target_classes

    def execute(self):
        camera_service = self.service
        net, classes = camera_service.Load_Yolo_model()
        blob = camera_service.prepare_frame(self.frame)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())
        
        boxes, confidences, class_ids = camera_service.process_predictions(outs, self.frame, self.target_classes)
        camera_service.bounding_boxes(self.frame, boxes, confidences, class_ids, classes)
        cv2.imshow("Detected Objects", self.frame)
        