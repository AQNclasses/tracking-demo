#! /usr/bin/python
# Alli N 2023

# contains multithread for getting video frames, possible future parallelization by splitting video

import cv2
from threading import Thread
from queue import Queue
from datetime import datetime

class VideoGet:
    """
    Continuously grabs frames from a VideoCapture object, dedicated thread.
    """

    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.w, self.h = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH), self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.FPS = self.stream.get(cv2.CAP_PROP_FPS)
        self.frame_num = round(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames = Queue(maxsize=50)
        self.FULL = False
        if self.stream.isOpened():
            print(f"Opened video {src} with {self.frame_num} {self.w}x{self.h} frames at {self.FPS} FPS")
        else:
            print(f"Couldn't find {src}")
            return False, [], [], []

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                if not self.FULL:
                    try:
                        self.grabbed, frame = self.stream.read()
                        self.frames.put(frame, block=False)
                    except:
                        self.FULL = True

    def next_frame(self):
        self.FULL = False
        self.frame = self.frames.get()
        return self.frame

    def stop(self):
        self.stopped = True

class VideoShow:
    """
    Show video frames using dedicated thread
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True

# Performance Profiling
class CountsPerSec:
    """
    Class that tracks the number of occurrences ("counts") of an
    arbitrary event and returns the frequency in occurrences
    (counts) per second. The caller must increment the count.
    """

    def __init__(self):
        self._start_time = None
        self._num_occurrences = 0

    def start(self):
        self._start_time = datetime.now()
        return self

    def increment(self):
        self._num_occurrences += 1

    def countsPerSec(self):
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        return self._num_occurrences / elapsed_time
