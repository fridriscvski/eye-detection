import cv2
import dlib
import numpy as np
from source.utils import Utils
from source.presenter import Presenter

def main():
    # setup
    presenter = Presenter()
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")

    # super loop
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)

            # Get blinking
            left_eye_ratio = Utils.get_blinking_ratio(frame, [36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = Utils.get_blinking_ratio(frame, [42, 43, 44, 45, 46, 47], landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

            if blinking_ratio > 5.7:
                presenter.presentText("Blinking", frame)

            else:
                # Get direction
                gaze_ratio_left_eye = Utils.get_gaze_ratio(frame, [36, 37, 38, 39, 40, 41], landmarks)
                gaze_ratio_right_eye = Utils.get_gaze_ratio(frame, [42, 43, 44, 45, 46, 47], landmarks)
                gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

                if gaze_ratio <= 0.8:
                    presenter.presentText("Right", frame)

                elif gaze_ratio > 1.2:
                    presenter.presentText("Left", frame)

        presenter.presentFrame(frame, "Frame")

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()