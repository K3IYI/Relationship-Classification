import argparse
import cv2
import numpy as np
import time
from pathlib import Path
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from take_snapshot import TakeSnapshot
from load_frame import LoadFrame
from detection import Detection

# Model is from https://github.com/yu4u/age-gender-estimation
pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'


def get_args():
    parser = argparse.ArgumentParser(description="This script capture image from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--video", type=str, default="0",
                        help="state the video input for web cam")
    parser.add_argument("--save_dir", type=str, default=".",
                        help="choose the directory to save image")
    parser.add_argument("--img_dir", type=str, default=None,
                        help="choose the directory to load image for detection")
    parser.add_argument("--no_snapshot", dest="snapshot", action="store_false",
                        help="skip snapshot to do detection")
    parser.add_argument("--live", dest="live", action="store_true",
                        help="use live stream for detection")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    depth = args.depth
    k = args.width
    weight_file = args.weight_file
    margin = args.margin
    video_input = args.video
    save_dir = Path(args.save_dir)
    img_dir = args.img_dir
    snapshot = args.snapshot
    live = args.live

    gender_th = 0.5
    tracker = 0
    last_stop = tracker
    not_detected = True

    if not weight_file:
        weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                               file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

    # load model and weights for gender detection
    img_size = 64
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)
    
    detector = Detection(gender_th)
    frame_loader = LoadFrame()

    # live detection
    if live:
        cap = frame_loader.live_cam(video_input)

        while True:
            is_read, frame = cap.read()

            if not is_read:
                print("*****No Input")
                time.sleep(0.1)
                continue

            # face detection
            number_of_person = detector.face_detection(frame, img_size)

            # if face detected
            # do age-gender and relationship detection
            if number_of_person > 0:
                detector.age_gender_detection(margin, model)
                detector.relationship_detection()

            cv2.imshow("result", frame)
            key = cv2.waitKey(1)

            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        # to take snapshot before detection
        if snapshot:
            TakeSnapshot().take_snapshot(video_input, save_dir)

        # load images to be detected
        if img_dir is not None:
            img_list = frame_loader.load_images_from_dir(Path(img_dir))
            if not Path(img_dir).exists():
                print("*****Non existing file/directory for detection")
        elif not save_dir.exists():
            print("*****Non existing file/directory for detection")
        else:
            img_list = frame_loader.load_images_from_dir(save_dir)

        # image detection
        while tracker in range(0, len(img_list)):
            img = img_list[tracker]

            if not_detected:
                # face detection
                number_of_person = detector.face_detection(img, img_size)

                # if face detected
                # do age-gender and relationship detection
                if number_of_person > 0:
                    detector.age_gender_detection(margin, model)
                    detector.relationship_detection()
                
                last_stop = tracker

            cv2.imshow("result", img)
            key = cv2.waitKey(-1)

            if key == ord("b"):
                if tracker is 0:
                    continue
                tracker -= 1
                not_detected = False
            elif key == ord('q'):
                break
            else:
                tracker += 1
                if tracker is last_stop + 1:
                    not_detected = True
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()