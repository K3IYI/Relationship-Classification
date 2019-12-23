import cv2
from pathlib import Path

class LoadFrame:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h


    def live_cam(self, source):
        try:
            cap = cv2.VideoCapture(int(source))
        except:
            cap = cv2.VideoCapture(source)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        return cap


    def load_images_from_dir(self, dir):
        img_list = []
        save_dir = Path(dir)

        for img_path in save_dir.glob("*.*"):
            img = cv2.imread(str(img_path), 1)

            if img is not None:
                h, w, _ = img.shape
                r = 640 / max(w, h)
                img_list.append(cv2.resize(img, (int(w * r), int(h * r))))
        return img_list