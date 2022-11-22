import cv2
import dlib
import numpy as np
from source.utils import Utils
from source.presenter import Presenter
from math import hypot

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

            # Create a frame around the interested eye - left
            roi = Utils.get_right_frame(frame, landmarks)
            altura, largura, _ = roi.shape
            Utils.draw_frames(roi, int(altura), int(largura))

            # TODO: - Apagar posteriormente. Não há necessidade de redimensionar esse frame
            rows, cols, _ = roi.shape
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # TODO: - Apagar posteriormente. Não há necessidade de desenhar essas linhas. Elas são apenas de teste
            # cv2.line(roi,(0,int(rows/2)),(cols,int(rows/2)),(255,0,0),1)
            # cv2.line(roi,(int(cols/2), 0),(int(cols/2), rows),(255,0,0),1)

            # Cria threshold para consideração apenas das partes mais escuras
            _, threshold = cv2.threshold(gray_roi, 35, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse = True)

            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.line(roi,(x+int(w/2),0),(x+int(w/2), rows),(0,255,0),1)
                cv2.line(roi, (0,y+int(h/2)), (cols, y+int(h/2)),(0,255,0),1)
                break

            # Get blinking
            left_eye_ratio = Utils.get_blinking_ratio(frame, [36, 37, 38, 39, 40, 41], landmarks)
            blinking_ratio = left_eye_ratio

            if blinking_ratio > 5.7 and blinking_ratio < 8:
                presenter.presentText("Blinking - " + str(blinking_ratio), frame)

            else:
                presenter.presentText("Not blinking - " + str(blinking_ratio), frame)
                

        presenter.presentFrame(roi, "Frame")

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

