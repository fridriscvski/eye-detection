import cv2

class Presenter():
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_PLAIN

    def presentText(self, text, frame):
        cv2.putText(frame, text, (20, 30), self.font, 2, (0, 0, 255))

    def presentFrame(self, frame, title):
        cv2.imshow(title, frame)