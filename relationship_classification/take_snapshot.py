import cv2
import time
from load_frame import LoadFrame
from datetime import datetime

class TakeSnapshot:
    def __init__(self):
        pass

    
    def take_snapshot(self, video_input, save_dir):
        save_counter = 0
        
        cap = LoadFrame().live_cam(video_input)
        
        while True:
            is_read, frame = cap.read()
        
            if not is_read:
                print("*****No Input")
                time.sleep(0.1)
                continue

            cv2.imshow("image", frame)
            waitKey = cv2.waitKey(1)

            current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            save_img_name = current_time + "-" + str(save_counter) + ".jpg"

            if waitKey == ord("s"):
                if not save_dir.exists():
                    save_dir.mkdir()
                cv2.imwrite(filename=str(save_dir/save_img_name), img=frame)
                save_counter += 1
            elif waitKey == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()