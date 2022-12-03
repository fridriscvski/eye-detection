from math import hypot
import cv2
import numpy as np

class Utils:
    eye_frame_cutted = False
    right_eye_cord = None

    def get_blinking_ratio(self, frame, eye_points, facial_landmarks):
        left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = self.__midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = self.__midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
        cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

        hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

        ratio = hor_line_lenght / ver_line_lenght
        return ratio

    def draw_frames(self, frame, altura, largura):
            Utils.draw_rectangle(frame, 0, 0, int((largura/3)), int(altura))
            Utils.draw_rectangle(frame, int(2*largura/3), 0, int(largura), int(altura))
            Utils.draw_rectangle(frame, int(largura/3), 0, int(2*largura/3), int(altura/3))
            Utils.draw_rectangle(frame, int(largura/3),int(2*altura/4),int(2*largura/3),int(altura))

    def draw_rectangle(self, frame, x_start, y_start, x_end, y_end):
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255,0,0), 2)

    def get_gaze_ratio_hor(self, frame, eye_points, facial_landmarks):
        left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                    (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                    (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                    (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                    (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                    (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)

        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])

        gray_eye = eye[min_y: max_y, min_x: max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        height, width = threshold_eye.shape
        left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        left_side_white = cv2.countNonZero(left_side_threshold)

        right_side_threshold = threshold_eye[0: height, int(width / 2): width]
        right_side_white = cv2.countNonZero(right_side_threshold)

        if left_side_white == 0:
            gaze_ratio = 1
        elif right_side_white == 0:
            gaze_ratio = 5
        else:
            gaze_ratio = left_side_white / right_side_white
        return gaze_ratio

    def get_gaze_ratio_ver(self, frame, eye_points, facial_landmarks):
        left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                    (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                    (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                    (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                    (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                    (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)

        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])

        gray_eye = eye[min_y: max_y, min_x: max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        height, width = threshold_eye.shape
        top_side_threshold = threshold_eye[0: int(height / 2), 0: width]
        top_side_white = cv2.countNonZero(top_side_threshold)

        bottom_side_threshold = threshold_eye[int(height / 2): height, 0: width]
        bottom_side_white = cv2.countNonZero(bottom_side_threshold)

        if top_side_white == 0:
            gaze_ratio = 1
        elif bottom_side_white == 0:
            gaze_ratio = 5
        else:
            gaze_ratio = top_side_white / bottom_side_white
        return gaze_ratio

    def get_right_frame(self, frame, facial_landmarks):    
        if not self.eye_frame_cutted:
            self.eye_frame_cutted = True
            self.right_eye_cord = {
                'x_max': facial_landmarks.part(36).x - 15,
                'x_min': facial_landmarks.part(39).x + 15,
                'y_max': facial_landmarks.part(38).y - 10,
                'y_min': facial_landmarks.part(41).y + 10
            }

        # Create a frame around the interested eye - left
        eye_frame = frame[self.right_eye_cord['y_max']: self.right_eye_cord['y_min'], self.right_eye_cord['x_max']: self.right_eye_cord['x_min']] 
        return cv2.flip(eye_frame, 1)

    def __midpoint(self, p1 ,p2):
        return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)