# -*- coding: utf-8 -*-
"""ベーシックなARマーカー認識モジュール。
"""
import cv2
import numpy as np
from .. import marker


class Marker(object):
    """This includes some properties of a marker.
    """
    
    def __init__(self, name, rotation, img, square):
        self.name = name
        self.rotation = rotation
        self.img = img
        self.square = square
        self.center = np.average(self.square, axis=0)
        # self.width = np.average(map(lambda ps: np.linalg.norm(ps), [
        #     square[0] - square[1], square[1] - square[2],
        #     square[2] - square[3], square[3] - square[0],
        # ]))


class ARMarkerDetector(object):
    
    def __init__(self, marker_files, size=(128, 128), threshold=0.90):
        """
        :param marker_files: マーカー名とマーカーファイルパスを辞書型で列挙。
        """
        detector = marker.TemplateDetector(size, threshold)
        for name, file in marker_files.items():
            img = cv2.imread(file) if type(file) is str else file
            detector.templates[name] = img
        detector.prepare()
        self.detector = detector
    
    def detect(self, src_img, draw=False):
        bin_img, _ = marker.preprocess_image(src_img)
        squares = marker.detect_squares(bin_img)
        markers = []
        for square in squares:
            tmp_img = marker.extract_image(bin_img, square)
            result = self.detector.detect(tmp_img)
            if result is not None:
                name, rotation = result
                markers.append(Marker(name, rotation, tmp_img, square))
        # 検出したマーカーを赤線で囲む
        if draw:
            cv2.drawContours(
                src_img, [m.square for m in markers], -1, (0, 0, 255), 2)
        return markers
            
            


