# -*- coding: utf-8 -*-
"""OpenCV utilities.
"""
import sys
import time
import os.path
import argparse
import cv2


class ArgumentParserError(Exception):
    pass

class ThrowingArgumentParser(argparse.ArgumentParser):
    """ArgumentParser throwing an error instead of exiting on a parsing error.
    
    .. seealso::
        http://stackoverflow.com/q/14728376
    """
    def error(self, message):
        raise ArgumentParserError(message)


class Window(object):
    """OpenCV window wrapper class.
    """
    
    ESCAPE = 27
    
    def __init__(self, name):
        self.name = name
        cv2.namedWindow(name)
    
    def update(self, img):
        """Update content image.
        """
        self.img = img
        cv2.imshow(self.name, img)
    
    def destroy(self):
        """Destroy window.
        """
        cv2.destroyWindow(self.name)
    
    def save(self, directory="img", filename=None, silent=False):
        """Save content image.
        """
        if filename is None:
            now = ("%.2f" % time.time()).replace(".", "_")
            filename = "%s-%s.png" % (name, now)
        path = os.path.join(directory, filename)
        cv2.imwrite(path, self.img)
        if not silent:
            print "%s is saved." % path
    
    @classmethod
    def wait(self, interval_ms):
        """Wrapper of ``cv2.waitKey``.
        """
        return cv2.waitKey(int(interval_ms))


class Capture(object):
    """This class defines the interface and provides factory method for users.
    """
    
    @classmethod
    def setup_argparser(cls, parser=None):
        """
        :param parser: Set ArgumentParser instance if you want to use your own ArgumentParser.
        :type parser: argparser.ArgumentParser
        """
        parser = parser or argparse.ArgumentParser()
        group = parser.add_argument_group("capturing media", "One of the following options is required.")
        exg = group.add_mutually_exclusive_group(required=True)
        exg.add_argument("-v", "--video", metavar="VIDEO_FILE")
        exg.add_argument("-i", "--image", metavar="IMAGE_FILE")
        exg.add_argument("-c", "--camera", nargs="?", type=int, const=0, metavar="DEVICE_ID=0")
        return parser
    
    @classmethod
    def create(cls, parsed_args=None):
        """
        :param parsed_args: If you use your own ArgumentParser, the parsed result is set here.
        """
        args = parsed_args or cls.setup_argparser().parse_args()
        if args.image is not None:
            return ImageCapture(args.image)
        elif args.video is not None:
            return VideoCapture(args.video)
        elif args.camera is not None:
            return VideoCapture(args.camera)
        assert "Come here? It's fatal error!"
    
    def get(self):
        """
        :return: The current frame.
        """
        raise NotImplementedError()
    
    @property
    def width(self):
        """Width of a frame.
        """
        raise NotImplementedError()
    
    @property
    def height(self):
        """Height of a frame.
        """
        raise NotImplementedError()
    
    @property
    def fps(self):
        """FPS (frame per second) of the media.
        """
        raise NotImplementedError()
        

class ImageCapture(Capture):
    
    def __init__(self, filename):
        img = cv2.imread(filename)
        if img is None:
            raise IOError("Cannot read file '{0}'".format(filename))
        self.img = img
    
    def get(self):
        return self.img
    
    @property
    def width(self):
        return self.img.shape[1]

    @property
    def height(self):
        return self.img.shape[0]
    
    @property
    def fps(self):
        return 0


class VideoCapture(Capture):
    
    def __init__(self, video=0):
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise IOError("Video '{0}' cannot be opened.".format(video))
        self.cap = cap
        
        # If no window is created, a video capture doesn't work.
        cv2.namedWindow("dummy")
        cv2.waitKey(1)
        cv2.destroyWindow("dummy")
        
    def get(self):
        got, img = self.cap.read()
        return (img if got else None)
    
    @property
    def width(self):
        return int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    
    @property
    def height(self):
        return int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    
    @property
    def fps(self):
        return int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        
        
        
