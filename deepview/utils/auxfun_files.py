
#
# import skimage.color
# from skimage import io
# from skimage.util import img_as_ubyte
# import cv2
import datetime
import numpy as np
import os
import subprocess
import warnings
import csv


# more videos are in principle covered, as OpenCV is used and allows many formats.
# SUPPORTED_VIDEOS = "avi", "mp4", "mov", "mpeg", "mpg", "mpv", "mkv", "flv", "qt", "yuv"
SUPPORTED_FILES = "csv", "txt", "doc", "xsls"


class fileReader:
    def __init__(self, video_path):
        if not os.path.isfile(video_path):
            raise ValueError(f'Video path "{video_path}" does not point to a file.')
        self.video_path = video_path
        self.video = csv.reader(video_path)
        # if not self.video.isOpened():
        #     raise IOError("Video could not be opened; it may be corrupted.")
        # self.parse_metadata()  # define width, height, #.frames, fps
        self._bbox = 0, 1, 0, 1
        self._n_frames_robust = None


    def get_bbox(self, relative=False):
        x1, x2, y1, y2 = self._bbox
        # if not relative:
        #     x1 = int(self._width * x1)
        #     x2 = int(self._width * x2)
        #     y1 = int(self._height * y1)
        #     y2 = int(self._height * y2)
        return x1, x2, y1, y2