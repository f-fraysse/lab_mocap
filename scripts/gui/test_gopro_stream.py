import cv2, time

class GoProCam:
    def __init__(self, index: int, size=(1920,1080), fps=30, fourcc='MJPG', warmup_frames=45):
        self.index = index
        self.size = size
        self.fps = fps
        self.fourcc = fourcc
        self.warmup_frames = warmup_frames
        self.cap = None

    def open(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
        if not cap.isOpened():
            raise RuntimeError(f"MSMF couldn't open camera at index {self.index}")

        # Ask for format/size/fps (backend may clamp silently)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Keep latency low (not always honored on MSMF, but helps if supported)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Warm-up to skip splash/black frames
        ok, frame = False, None
        for _ in range(self.warmup_frames):
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)

        if not ok:
            cap.release()
            raise RuntimeError("Opened but didn't deliver frames (try different FOURCC or 720p).")

        self.cap = cap
        return True

    def read(self):
        if self.cap is None:
            raise RuntimeError("Camera not opened")
        return self.cap.read()

    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None

if __name__ == "__main__":
    cam = GoProCam(index=0, size=(1920,1080), fps=30, fourcc='MJPG', warmup_frames=20)
    cam.open()
    t0, frames = time.time(), 0
    while True:
        ok, frame = cam.read()
        if not ok: break
        frames += 1
        cv2.imshow("GoPro MSMF 1080p MJPG", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.close()
    cv2.destroyAllWindows()
