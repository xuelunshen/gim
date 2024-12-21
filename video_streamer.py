import math

from pathlib import Path
from torchvision.io import VideoReader


class VideoStreamer:
    """ Class to help process image streams. Four types of possible inputs:"
        1.) USB Webcam.
        2.) An IP camera
        3.) A directory of images (files in directory matching 'image_glob').
        4.) A video file, such as an .mp4 or .avi file.
    """
    def __init__(self, basedir, resize, df, skip, vrange=None, image_glob=None, max_length=1000000):
        """
        The function takes in a directory, a resize value, a skip value, a glob value, and a
        max length value.

        The function then checks if the directory is a number, if it is, it sets the cap to
        a video capture of the directory.

        If the directory starts with http or rtsp, it sets the cap to a video capture of the
        directory.

        If the directory is a directory, it sets the listing to a list of the directory.

        If the directory is a file, it sets the cap to a video capture of the directory.

        If the directory is none of the above, it raises a value error.

        If the directory is a camera and the cap is not opened, it raises an IO error.

        Args:
          basedir: The directory where the images or video file are stored.
          resize: The size of the image to be returned.
          df: The frame rate of the video.
          skip: This is the number of frames to skip between each frame that is read.
          vrange: Video time range
          image_glob: A list of glob patterns to match the images in the directory.
          max_length: The maximum number of frames to read from the video. Defaults to
        1000000
        """
        if vrange is None:
            vrange = [0, -1]

        self.listing = []
        self.skip = skip

        if Path(basedir).exists():
            self.video = VideoReader(basedir, 'video')
            meta = self.video.get_metadata()
            seconds = math.floor(meta['video']['duration'][0])
            self.fps = int(meta['video']['fps'][0])
            start, end = max(0, vrange[0]), min(seconds, vrange[1])
            end = seconds if end == -1 else end
            assert start < end, 'Invalid video range'
            self.range = [start, end]
            self.listing = range(start*self.fps, end*self.fps+1)
            self.listing = self.listing[::self.skip]

        else:
            raise ValueError('VideoStreamer input \"{}\" not recognized.'.format(basedir))

    def __len__(self):
        return len(self.listing)

    def __getitem__(self, i):
        image = next(self.video.seek(i/self.fps))['data'].permute(1, 2, 0).numpy()
        return image
