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
        camera_service.complet_detection(self.frame, self.target_classes)
