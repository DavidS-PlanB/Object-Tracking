from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import sys
import cv2

class CameraServices:

    def track(self):
        video = cv2.VideoCapture(0)

        if not video.isOpened():
            print("Fehler: Video konnte nicht geöffnet werden.")
            sys.exit()

        # Wähle das erste Frame aus und markiere das zu verfolgende Objekt
        ok, frame = video.read()
        if not ok:
            print("Fehler: Frame konnte nicht gelesen werden.")
            sys.exit()

        # Rechteck (ROI) manuell im ersten Frame auswählen
        bbox = cv2.selectROI(frame, False)
        print("Gewähltes ROI:", bbox)

        # Tracker initialisieren
        tracker = cv2.TrackerCSRT_create() if hasattr(cv2, 'TrackerCSRT_create') else cv2.legacy.TrackerCSRT_create()

         # Du kannst hier einen anderen Tracker auswählen

        # Tracker mit dem ersten Frame und der ROI starten
        ok = tracker.init(frame, bbox)

        # Frame für Frame durchgehen
        while True:
            ok, frame = video.read()
            if not ok:
                break

            # Objekt verfolgen
            ok, bbox = tracker.update(frame)

            # Bounding-Box zeichnen, wenn das Tracking erfolgreich ist
            if ok:
                (x, y, w, h) = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Tracking failed", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # Frame anzeigen
            cv2.imshow("Tracking", frame)

            # Beenden mit der Taste 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_service = CameraServices()
    camera_service.track()
