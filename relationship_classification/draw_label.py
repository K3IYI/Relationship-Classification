import cv2

class DrawLabel:
    def __init__(self, image_height=None, font=cv2.FONT_HERSHEY_SIMPLEX,
                font_scale=0.8, thickness=1):
        self.image_height = image_height
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness


    def __get_feature(self, point, label):
        size = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)[0]
        x, y = point
        return size, x, y


    def draw_label_faces(self, image, point, label):
        size, x, y = self.__get_feature(point, label)
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, self.font, self.font_scale, (255, 255, 255), self.thickness, lineType=cv2.LINE_AA)


    def draw_label_number_of_person(self, image, point, label):
        size, x, y = self.__get_feature(point, label)
        cv2.rectangle(image, (x, y + size[1] + 8), (x + size[0] + 4, y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, (x, y + size[1] + 1), self.font, self.font_scale, (255, 255, 255), self.thickness, lineType=cv2.LINE_AA)


    def draw_label_relationship(self, image, point, label, image_height):
        size, x, y = self.__get_feature(point, label)
        cv2.rectangle(image, (x, y + image_height), (x + size[0] + 2, y + image_height - size[1] - 14), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, (x, y + image_height - 10), self.font, self.font_scale, (255, 255, 255), self.thickness, lineType=cv2.LINE_AA)