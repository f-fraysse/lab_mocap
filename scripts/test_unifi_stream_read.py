import cv2
import os
import time
import numpy as np
from rtmlib import RTMDet, RTMPose, draw_skeleton
from paths import MODEL_DIR, DATA_DIR, OUTPUT_VIDEO_DIR, OUTPUT_H5_DIR, ensure_output_dirs

ensure_output_dirs()

url_1 = "rtsp://ubnt:ubnt@192.168.5.41:554/s0"
url_2 = "rtsp://ubnt:ubnt@192.168.5.45:554/s0"
url_3 = "rtsp://ubnt:ubnt@192.168.5.42:554/s0"
url_4 = "rtsp://ubnt:ubnt@192.168.5.48:554/s0"

cap_1 = cv2.VideoCapture(url_1)
# cap_2 = cv2.VideoCapture(url_2)
# cap_3 = cv2.VideoCapture(url_3)
# cap_4 = cv2.VideoCapture(url_4)
cv2.setUseOptimized(True)

def stitch_frames(frame_1, frame_2, frame_3, frame_4):
    # Resize each frame to 960x540
    size = (960, 540)
    f1 = cv2.resize(frame_1, size)
    f2 = cv2.resize(frame_2, size)
    f3 = cv2.resize(frame_3, size)
    f4 = cv2.resize(frame_4, size)

    # Top row: f1 | f2
    top = np.hstack((f1, f2))
    # Bottom row: f3 | f4
    bottom = np.hstack((f3, f4))

    # Combine rows
    combined = np.vstack((top, bottom))

    return combined

while True:
    
    ret, frame_1 = cap_1.read()
    # ret, frame_2 = cap_2.read()
    # ret, frame_3 = cap_3.read()
    # ret, frame_4 = cap_4.read()
    cap_time = time.time()    
   

    #combine all 4 streams into single 
    # stitched_img = stitch_frames(frame_1, frame_2, frame_3, frame_4)

    # resize if single camera
    frame_1 = cv2.resize(frame_1, (1920, 1080))

    # Timing info
    cap_duration = (cap_time - start_time) * 1000

    frame = cv2.putText(frame_1, f'cap: {cap_duration: .2f} ms', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
   
    cv2.imshow("Camera", frame_1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_1.release()
# cap_2.release()
# cap_3.release()
# cap_4.release()
cv2.destroyAllWindows()