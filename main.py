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

            if blinking_ratio > 5.7 and blinking_ratio < 8:
                presenter.presentText("Blinking - " + str(blinking_ratio), frame)

            else:
                # Get direction
                gaze_ratio_left_eye = Utils.get_gaze_ratio_hor(frame, [36, 37, 38, 39, 40, 41], landmarks)
                gaze_ratio_right_eye = Utils.get_gaze_ratio_hor(frame, [42, 43, 44, 45, 46, 47], landmarks)
                gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

                if gaze_ratio <= 0.5:
                    presenter.presentText("Right - " + str(gaze_ratio), frame)

                elif gaze_ratio > 1.8:
                    presenter.presentText("Left - " + str(gaze_ratio), frame)

                else:
                    gaze_ratio_top_eye = Utils.get_gaze_ratio_ver(frame, [36, 37, 38, 39, 40, 41], landmarks)
                    gaze_ratio_bottom_eye = Utils.get_gaze_ratio_ver(frame, [42, 43, 44, 45, 46, 47], landmarks)
                    gaze_ratio = (gaze_ratio_top_eye + gaze_ratio_bottom_eye) / 2

                    if gaze_ratio <= 0.2:
                        presenter.presentText("top - " + str(gaze_ratio), frame)

                    elif gaze_ratio > 3:
                        presenter.presentText("bottom - " + str(gaze_ratio), frame)

        presenter.presentFrame(frame, "Frame")

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()