import cv2
import dlib
import numpy as np
from draw_label import DrawLabel

class Detection:
    def __init__(self, gender_th, face_detector=dlib.get_frontal_face_detector()):
        self.gender_th = gender_th
        self.face_detector = face_detector

        self.__img = None
        self.__img_size = None
        self.__img_h = None
        self.__img_w = None
        self.__detected = None
        self.__faces = None
        self.__number_of_person = None
        self.__predicted_genders = None
        self.__predicted_ages = None

        self.__draw_label = DrawLabel()


    def face_detection(self, img, img_size):
        self.__img = img
        self.__img_size = img_size

        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.__img_h, self.__img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        self.__detected = self.face_detector(input_img, 1)
        self.__faces = np.empty((len(self.__detected), self.__img_size, self.__img_size, 3))

        self.__number_of_person = len(self.__detected)

        number_of_person_label = "Detected: {}".format(self.__number_of_person)
        self.__draw_label.draw_label_number_of_person(img, (0, 0), number_of_person_label)
        
        return self.__number_of_person


    def age_gender_detection(self, margin, model):
        for i, d in enumerate(self.__detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), self.__img_w - 1)
            yw2 = min(int(y2 + margin * h), self.__img_h - 1)
            cv2.rectangle(self.__img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
            self.__faces[i, :, :, :] = cv2.resize(self.__img[yw1:yw2 + 1, xw1:xw2 + 1, :], (self.__img_size, self.__img_size))

        # predict ages and genders of the detected faces
        results = model.predict(self.__faces)
        self.__predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        self.__predicted_ages = results[1].dot(ages).flatten()

        for i, d in enumerate(self.__detected):
            age_grp = self.__determine_age_group(self.__predicted_ages[i])
            single_face_label = "{}, {}".format(age_grp,
                                    "M" if self.__predicted_genders[i][0] < self.gender_th else "F")
            self.__draw_label.draw_label_faces(self.__img, (d.left(), d.top()), single_face_label)


    def relationship_detection(self):
        age_gender_grp = self.__age_gender_group_classification()
        relationship = self.__determine_relationship(self.__number_of_person, age_gender_grp)

        relationship_label = "Relationship: {}".format(relationship)
        self.__draw_label.draw_label_relationship(self.__img, (0, 0), relationship_label, image_height=self.__img_h)


    def __determine_age_group(self, age):
        age_grp = ""
        if 0 < age <= 18:
            age_grp = "kid"
        elif 18 < age <= 45:
            age_grp = "adult"
        elif 45 < age:
            age_grp = "elderly"
        return age_grp


    def __age_gender_group_classification(self):
        age_gender_grp = {"kid_male":0, "kid_female":0, "adult_male":0,
                     "adult_female":0, "elderly_male":0, "elderly_female":0}
        
        for i, d in enumerate(self.__detected):
            age_grp = self.__determine_age_group(self.__predicted_ages[i])

            if age_grp is "kid":
                if self.__predicted_genders[i][0] < self.gender_th:
                    age_gender_grp["kid_male"] += 1
                elif self.__predicted_genders[i][0] > self.gender_th:
                    age_gender_grp["kid_female"] += 1
            elif age_grp is "adult":
                if self.__predicted_genders[i][0] < self.gender_th:
                    age_gender_grp["adult_male"] += 1
                elif self.__predicted_genders[i][0] > self.gender_th:
                    age_gender_grp["adult_female"] += 1
            elif age_grp is "elderly":
                if self.__predicted_genders[i][0] < self.gender_th:
                    age_gender_grp["elderly_male"] += 1
                elif self.__predicted_genders[i][0] > self.gender_th:
                    age_gender_grp["elderly_female"] += 1
        return age_gender_grp


    def __determine_relationship(self, number_of_person, age_gender_grp):
        relationship = "Unknown"

        is_female_kid = True if age_gender_grp["kid_female"] > 0 else False
        is_male_kid = True if age_gender_grp["kid_male"] > 0 else False
        is_both_gender_kid = True if age_gender_grp["kid_female"] > 0 and age_gender_grp["kid_male"] > 0 else False
        
        is_female_adult = True if age_gender_grp["adult_female"] > 0 else False
        is_male_adult = True if age_gender_grp["adult_male"] > 0 else False
        is_both_gender_adult = True if age_gender_grp["adult_female"] > 0 and age_gender_grp["adult_male"] > 0 else False
        
        is_female_elderly = True if age_gender_grp["elderly_female"] > 0 else False
        is_male_elderly = True if age_gender_grp["elderly_male"] > 0 else False
        is_both_gender_elderly = True if age_gender_grp["elderly_female"] > 0 and age_gender_grp["elderly_male"] > 0 else False

        def try_catch_relationship(is_female_kid, is_male_kid, is_both_gender_kid, is_female_adult,
                                    is_male_adult, is_both_gender_adult, is_female_elderly, is_male_elderly,
                                    is_both_gender_elderly):
            if (is_female_adult and is_female_kid) or (is_female_elderly and is_female_adult):
                relationship = "mother and daughter"
            elif (is_female_adult and is_male_kid) or (is_female_elderly and is_male_adult):
                relationship = "mother and son"
            elif (is_male_adult and is_female_kid) or (is_male_elderly and is_female_adult):
                relationship = "father and daughter"
            elif (is_male_adult and is_male_kid) or (is_male_elderly and is_male_adult):
                relationship = "father and son"
            elif is_female_elderly and is_female_kid:
                relationship = "grnadmother and granddaughter"
            elif is_female_elderly and is_male_kid:
                relationship = "grnadmother and grandson"
            elif is_male_elderly and is_female_kid:
                relationship = "grandfather and granddaughter"
            elif is_male_elderly and is_male_kid:
                relationship = "grandfather and grandson"
            else:
                relationship = "friends"
            return relationship

        if number_of_person >= 3:
            is_family1 = True if (is_female_kid or is_male_kid) and is_both_gender_adult else False
            is_family2 = True if (is_female_adult or is_male_adult) and is_both_gender_elderly else False
            if is_family1 or is_family2:
                relationship = "family, parents and child"
            elif (is_female_kid or is_male_kid) and is_both_gender_elderly:
                relationship = "family, grandparents and kids"
            else:
                relationship = try_catch_relationship(is_female_kid, is_male_kid, is_both_gender_kid, is_female_adult,
                                is_male_adult, is_both_gender_adult, is_female_elderly, is_male_elderly,
                                is_both_gender_elderly)
        elif number_of_person is 2:
            if is_male_adult and is_female_adult:
                relationship = "couple"
            else:
                relationship = try_catch_relationship(is_female_kid, is_male_kid, is_both_gender_kid, is_female_adult,
                                is_male_adult, is_both_gender_adult, is_female_elderly, is_male_elderly,
                                is_both_gender_elderly)
        else:
            relationship = ":o"
        return relationship