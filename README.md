# Object-Tracking

Feature-Idee:

Bestenliste wer am schnellsten oder genausten das Glas befühlt hat

#Für Jetson Tx2
Das normale Videocapture muss durch das ausgetauscht werden:

    # GStreamer-Pipeline für NVIDIA Jetson verwenden
    pipeline = (
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
        "nvvidconv flip-method=2 ! video/x-raw, width=640, height=480, format=BGRx ! videoconvert ! appsink"
    )

    # OpenCV VideoCapture mit GStreamer-Pipeline öffnen
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
