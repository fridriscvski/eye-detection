import cv2

class Presenter():
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_TRIPLEX

    def presentText(self, text, frame):
        cv2.putText(frame, text, (50, 100), self.font, 2, (0, 0, 255), 3) 

    def presentFrame(self, frame, title):
        cv2.imshow(title, frame)