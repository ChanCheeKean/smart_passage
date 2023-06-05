import threading
import cv2
from utils.config_loader import config
### video capture ###
class FreshestFrame(threading.Thread):
    '''return the last frame from the camera input'''
    def __init__(self, capture, name='FreshestFrame'):
        self.capture = capture
        assert self.capture.isOpened()
        # this lets the read() method block until there's a new frame
        self.cond = threading.Condition()
        # this allows us to stop the thread gracefully
        self.running = False
        # keeping the newest frame around
        self.frame = None
        # passing a sequence number allows read() to NOT block
        # if the currently available one is exactly the one you ask for
        self.latestnum = 0
        # this is just for demo purposes        
        self.callback = None
        super().__init__(name=name)
        self.start()

    def start(self):
        self.running = True
        super().start()

    def release(self, timeout=None):
        self.running = False
        self.join(timeout=timeout)
        self.capture.release()

    def run(self):
        counter = 0
        while self.running:
            # block for fresh frame
            (rv, img) = self.capture.read()
            assert rv
            counter += 1
            # publish the frame
            with self.cond: # lock the condition for this operation
                self.frame = img if rv else None
                self.latestnum = counter
                self.cond.notify_all()
            if self.callback:
                self.callback(img)

    def read(self, wait=True, seqnumber=None, timeout=None):
        # with no arguments (wait=True), it always blocks for a fresh frame
        # with wait=False it returns the current frame immediately (polling)
        # with a seqnumber, it blocks until that frame is available (or no wait at all)
        # with timeout argument, may return an earlier frame;
        # may even be (0, None) if nothing received yet
        with self.cond:
            if wait:
                if seqnumber is None:
                    seqnumber = self.latestnum+1
                if seqnumber < 1:
                    seqnumber = 1
                rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
                if not rv:
                    return (self.latestnum, self.frame)
            return (self.latestnum, self.frame)
        
class ImageLoader(object):
    def __init__(self):
        video_config = config['video']
        self.cnt = 0
        self.source = video_config['source']
        self.save_video = video_config['save_video']
        self.plot_roi = video_config['plot_roi']
        self.output_file_name = video_config['output_file_name']
        self.img_sz = video_config['img_sz']
        self.font_size = video_config['font_size']
        self.mm_per_pixel = video_config['font_size']
        self.trigger_distance = video_config['trigger_distance']
        self.out_writter = cv2.VideoWriter(
            self.output_file_name, 
            cv2.VideoWriter_fourcc('M','J','P','G'), 
            30, 
            self.img_sz)
        
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.startswith('rtsp')
        self.video_loader = cv2.VideoCapture(self.source)
        self.id_paid = []
        self.id_complete = []
        self.passenger_count = 0

        # skip some frames and jump
        self.frame_num = self.video_loader.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_loader.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num - 600)
        
        if self.webcam:
            self.video_loader = FreshestFrame(self.video_loader)
        
    def __del__(self):
        self.video_loader.release()

    def get_frame(self):

        if self.webcam:
            self.cnt, image = self.video_loader.read(seqnumber=self.cnt+1)
        else:
            self.cnt, image = self.video_loader.read()

        return self.cnt, image