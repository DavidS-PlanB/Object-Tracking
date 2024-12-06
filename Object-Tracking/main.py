import cv2
from Object_Tracking_Commands.Commands.Command import Command
from Object_Tracking_Core.Services.Camera_Service import CameraService
from Object_Tracking_Commands.Commands.DetectObjectsCommand import DetectObjectsCommand
from Object_Tracking_Commands.Commands.CameraInvoker import CameraInvoker

def main():
    cap = cv2.VideoCapture(0) #0 default port Camera

    if not cap.isOpened():
        print("Error: Camera could not be oppend.")
        return 

    camera_service = CameraService()
    invoker = CameraInvoker()
    target_classes = None #[39, 41]
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        # Command to create objects
        detect_command = DetectObjectsCommand(camera_service, frame, cap, target_classes= None)
        invoker.set_command(detect_command)

        # Execute the command
        invoker.execute_commands()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera_service.stop_tracking(cap)

if __name__ == "__main__":
    main()
