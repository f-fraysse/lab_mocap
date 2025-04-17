import cv2
import os
import time
import numpy as np
from rtmlib import RTMDet, RTMPose, draw_skeleton
from paths import MODEL_DIR, DATA_DIR, OUTPUT_VIDEO_DIR, OUTPUT_H5_DIR, ensure_output_dirs
from threading import *

ensure_output_dirs()

use_threaded = True

url_1 = "rtsp://192.168.1.48:8554/file_example_WEBM_480_900KB.webm"
url_2 = "rtsp://192.168.1.48:8554/file_example_WEBM_480_900KB_v2.webm"
url_3 = "rtsp://192.168.1.48:8554/file_example_WEBM_480_900KB.webm"
url_4 = "rtsp://192.168.1.48:8554/file_example_WEBM_480_900KB_v2.webm"

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


class ThreadedVideoCapture:
    
    def __init__(self, url, barrier):
        self.bar = barrier
        self.vcap = cv2.VideoCapture(url)
        
        if self.vcap.isOpened() is False :
            print("Error accessing webcam stream." + url)
            exit(0)
            
        # init the camera
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print("No more frames to read")
            exit(0)
            
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
        self.running = False
                    
    def start_thread(self):
        self.running = True
        self.t.start()
                    
    def stop(self):
        self.running = False                         
                    
    def update(self):    
        while True:
            if not self.running:
                break            
            self.grabbed, self.frame = self.vcap.read()
            self.bar.wait()            
            if self.grabbed is False:
                self.stop()
        self.vcap.release()
            
        

if __name__ == "__main__":
    
    cv2.setUseOptimized(True)
    
    if use_threaded:
    
        bar = Barrier(5)
        
        tv1 = ThreadedVideoCapture(url_1, bar)
        tv2 = ThreadedVideoCapture(url_2, bar)
        tv3 = ThreadedVideoCapture(url_3, bar)
        tv4 = ThreadedVideoCapture(url_4, bar)
        
        tv1.start_thread()
        tv2.start_thread()
        tv3.start_thread()
        tv4.start_thread()
                
    else:
        cap_1 = cv2.VideoCapture(url_1)
        cap_2 = cv2.VideoCapture(url_2)
        cap_3 = cv2.VideoCapture(url_3)
        cap_4 = cv2.VideoCapture(url_4)
    
    
    while True:
        
        start_time = time.perf_counter()
        
        if use_threaded:
            
            bar.wait() # wait for all threads to complete frame capture before proceeding on the main loop
            
            frame_1 = tv1.frame
            frame_2 = tv2.frame
            frame_3 = tv3.frame
            frame_4 = tv4.frame
        else:    
            ret, frame_1 = cap_1.read()
            ret, frame_2 = cap_2.read()
            ret, frame_3 = cap_3.read()
            ret, frame_4 = cap_4.read()
        
        cap_time = time.perf_counter()    
    
    
        #combine all 4 streams into single 
        stitched_img = stitch_frames(frame_1, frame_2, frame_3, frame_4)
    
        # resize if single camera
        frame_1 = cv2.resize(stitched_img, (1920, 1080))
    
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