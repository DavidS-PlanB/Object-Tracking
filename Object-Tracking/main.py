import cv2
from Object_Tracking_Core.Services.Camera_Service import CameraService
from Object_Tracking_Commands.Commands.DetectObjectsCommand import DetectObjectsCommand
from Object_Tracking_Commands.Commands.CameraInvoker import CameraInvoker
from Object_Tracking_Core.Services.SmartDownloader import SmartDownloader
from Object_Tracking_Commands.Commands.SmartDownload_Command import SmartDownload_Command

def main():
    cap = cv2.VideoCapture(0) #0 default port Camera

    if not cap.isOpened():
        print("Error: Camera could not be oppend.")
        return 

    invoker = CameraInvoker()

    smartDonwloader_service = SmartDownloader()
    smartDonwloader_command = SmartDownload_Command(smartDonwloader_service)
    invoker.set_command(smartDonwloader_command)
    invoker.execute_commands()

    camera_service = CameraService()

    target_classes = [39, 41] #None für kein Filter
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        # Command to create objects

        detect_command = DetectObjectsCommand(camera_service, frame, cap, target_classes)
        invoker.set_command(detect_command)

        # Execute the command
        invoker.execute_commands()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera_service.stop_tracking(cap)

if __name__ == "__main__":
    main()
