import cv2

class VideoProcessor:
    def __init__(self, video_path, **kwargs):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.id = 0
        self.data = kwargs

    @property
    def width(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    @property
    def height(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)               

    @property
    def duration(self):
        cap = self.cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return frame_count/fps

    @property
    def pos(self):
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    @pos.setter
    def pos(self, pos):
        self.id = pos
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos) 

    @property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)
            
    @property
    def frame_count(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __str__(self):
        return self.info()

    def info(self):
        info = "file: " + self.video_path
        info += "\n" +str(self.width) + "x" + str(self.height)        
        info += '\nfps = ' + str(self.fps)
        info += '\nnumber of frames = ' + str(self.frame_count)

        duration = self.duration
        info += '\nduration (S) = ' + str(duration)
        minutes = int(duration/60)
        seconds = duration%60
        info += '\nduration (M:S) = ' + str(minutes) + ':' + str(seconds)
        return info    

    def images(self, start=0, end=None):
        self.pos = start
        success, image = self.cap.read()
        while success:
            if end is not None and self.id == end:
                yield image
                return

            yield image
            self.id += 1            
            success, image = self.cap.read()

    @property
    def image(self):
        _, image = self.cap.read()
        self.id += 1
        return image                    

    def iterate(self, process, start=0, end=None):
        last = self.frame_count - 1 if end is None else end 
        for image in self.images(start=start, end=end):
            if process(image=image, pos=self.id, video=self, last=last, **self.data):
                break