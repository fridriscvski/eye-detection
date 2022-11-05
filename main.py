import cv2
import numpy as np
import dlib
from math import hypot

def pupil_detection():

    cap = cv2.VideoCapture("eye_recording.flv")

    while True:

        ret, frame = cap.read()
        if ret is False:
            break
        roi = frame
        rows, cols, _ = roi.shape
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.GaussianBlur(gray_roi, (7,7),0)

        _, threshold = cv2.threshold(gray_roi, 5, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse = True)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect (cnt)
            #cv2.drawContours(roi, [cnt], -1, (0,0,255),3)
            cv2.rectangle(roi,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.line(roi,(x+int(w/2),0),(x+int(w/2), rows),(0,255,0),2)
            cv2.line(roi, (0,y+int(h/2)), (cols, y+int(h/2)),(0,255,0),2)
            break

        cv2.imshow("threshold", threshold)
        cv2.imshow("Roi", gray_roi)
        cv2.imshow("Roi2", roi)
        key = cv2.waitKey(30)
        if key == 27:
            break

    cv2.destroyAllWindows()


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def midpoint(p1,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y))

font = cv2.FONT_HERSHEY_COMPLEX

def eye_long_detect():

    # inicia captura de video
    cap = cv2.VideoCapture(0)

    # cria detector e previsor de faces e de partes do rosto usando a bilbioteca
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    _, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:

    # com o previsor, detecto os pontos do rosto
        landmarks = predictor(gray, face)
        right_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
    
        print(right_eye_ratio)
        if right_eye_ratio > 3:
            print("piscou")

    # Cria o dicionário apenas com os pontos que precisamos - 4 extremos do olho
        right_eye_cord = {
            'x_max': landmarks.part(36).x - 15,
            'x_min': landmarks.part(39).x + 15,
            'y_max': landmarks.part(38).y - 5,
            'y_min': landmarks.part(41).y + 5
        }

    while True:
        # captura a face do usuario
        _, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        

        # Draw eye track point
        # cv2.circle(frame, (landmarks.part(36).x, landmarks.part(36).y), 4, (255, 0, 0), 1)
        # cv2.circle(frame, (landmarks.part(37).x, landmarks.part(37).y), 4, (255, 0, 0), 1)
        # cv2.circle(frame, (landmarks.part(38).x, landmarks.part(38).y), 4, (255, 0, 0), 1)
        # cv2.circle(frame, (landmarks.part(39).x, landmarks.part(39).y), 4, (255, 0, 0), 1)
        # cv2.circle(frame, (landmarks.part(40).x, landmarks.part(40).y), 4, (255, 0, 0), 1)
        # cv2.circle(frame, (landmarks.part(41).x, landmarks.part(41).y), 4, (255, 0, 0), 1)

        # cria o retângulo apenas no olho para fazermos o tratamento apenas nesse quadrado
        cv2.rectangle(frame, (right_eye_cord['x_max'], right_eye_cord['y_max']),  (right_eye_cord['x_min'], right_eye_cord['y_min']),  (0, 255, 0), 2)
        frame = frame[right_eye_cord['y_max']: right_eye_cord['y_min'], right_eye_cord['x_max']: right_eye_cord['x_min']]  
        frame = cv2.flip(frame, 1)

        roi = cv2.resize(frame, (900, 600), interpolation=cv2.INTER_AREA)
        rows, cols, _ = roi.shape
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        cv2.line(roi,(0,int(rows/2)),(cols,int(rows/2)),(255,0,0),1)
        cv2.line(roi,(int(cols/2), 0),(int(cols/2), rows),(255,0,0),1)
        

        # Cria threshold para consideração apenas das partes mais escuras
        _, threshold = cv2.threshold(gray_roi, 35, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse = True)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            # cv2.drawContours(roi, [cnt], -1, (0,0,255),3)
            # cv2.rectangle(roi,(x,y),(x+w,y+h),(255,0,0),2)
            print((0,y+int(h/2)), (cols, y+int(h/2)))

            cv2.line(roi,(x+int(w/2),0),(x+int(w/2), rows),(0,255,0),1)
            cv2.line(roi, (0,y+int(h/2)), (cols, y+int(h/2)),(0,255,0),1)
            
            break

        cv2.imshow("threshold", threshold)
        cv2.imshow("Roi", gray_roi)
        cv2.imshow("Roi2", roi)      
        

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break
    
    
eye_long_detect()


