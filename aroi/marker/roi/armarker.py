# -*- coding: utf-8 -*-
"""ARマーカーを使ったROI型ARモジュール。
"""
import cv2
import numpy as np
from .. import ar
from ... import geo
from ... import marker


class ROI(object):
    
    def __init__(self, marker_num_xy, tl=None, tr=None, br=None, bl=None):
        """
        :param marker_num_xy: (縦線数, 横線数)。格子の数が横方向に3個、縦方向に2個なら、(4, 3)
        """
        self.marker_num_xy = marker_num_xy
        self.set(tl, tr, br, bl)

    def set(self, tl, tr, br, bl):
        """各引数はピクセル(x, y)。
        """
        self.tl = tl
        self.tr = tr
        self.br = br
        self.bl = bl
    
    def get_points(self):
        return np.float32([self.tl, self.tr, self.br, self.bl])
    
    def get_points_for_estimation(self, roi_width=1):
        """位置推定のための点列を返す。
        
        :param roi_width: Each roi width(=height) in the virtual 3d space.
        """
        w = (self.marker_num_xy[0]-1) * roi_width * 0.5
        h = (self.marker_num_xy[1]-1) * roi_width * 0.5
        obj_points = np.float32(
            [[-w, 0, h], [w, 0, h], [w, 0, -h], [-w, 0, -h]])
        return self.get_points(), obj_points


class Detector(object):
    
    MARKER_NAMES = ["TL", "TR", "BR", "BL", "T", "R", "B", "L"]
    
    DEFAULT_MARKER_FILES = {
        "TL": "top_left.png",
        "TR": "top_right.png",
        "BR": "bottom_right.png",
        "BL": "bottom_left.png",
        "T": "top.png",
        "R": "right.png",
        "B": "bottom.png",
        "L": "left.png",
    }

    def __init__(self, marker_files, marker_num_xy):
        """
        :param marker_files: マーカーファイルの一覧。TL,TR,BR,BL,T,R,B,Lの要素を持つ辞書型オブジェクト。例としてDEFAULT_MARKER_FILESを参照。
        :param marker_num_xy: 格子（＝各辺におけるマーカー）が何個あるかを ``(横, 縦)`` で指定する。
        """
        self.detector = ar.ARMarkerDetector(
            marker_files or self.DEFAULT_MARKER_FILES, (32, 32), 0.85)
        self.marker_num_xy = marker_num_xy
    
    def detect(self, src_img, draw=False):
        """
        :param src_img: カメラからの生の入力画像。
        :param draw: デバッグ描画するかどうか。
        """
        markers = self.detector.detect(src_img, draw)
        # 4頂点を取得
        roi = self._get_roi(markers)
        if roi is not None and draw:
            cv2.drawContours(src_img,
                [np.int32([roi.tl, roi.tr, roi.br, roi.bl])],
                -1, (0, 0, 255), 1)
        return roi
    
    def _get_roi(self, markers):
        # マーカーの仕分け
        tl = tr = br = bl = None
        t = []; r = []; b = []; l = []
        for m in markers:
            if m.name == "TL": tl = m.center
            elif m.name == "TR": tr = m.center
            elif m.name == "BR": br = m.center
            elif m.name == "BL": bl = m.center
            elif m.name == "T": t.append(m.center)
            elif m.name == "R": r.append(m.center)
            elif m.name == "B": b.append(m.center)
            elif m.name == "L": l.append(m.center)
        
        # 角のマーカーがない時の補完処理。
        if tl is None:
            tl = self._get_intersection(tr, t, bl, l)
            if tl is None:
                return None
        if tr is None:
            tr = self._get_intersection(tl, t, br, r)
            if tr is None:
                return None
        if br is None:
            br = self._get_intersection(tr, r, bl, b)
            if br is None:
                return None
        if bl is None:
            bl = self._get_intersection(br, b, tl, l)
            if bl is None:
                return None

        return ROI(self.marker_num_xy, tl, tr, br, bl)
    
    def _get_intersection(self, corner1, side1, corner2, side2):
        ps1 = list(side1)
        if corner1 is not None:
            ps1.append(corner1)
        ps2 = list(side2)
        if corner2 is not None:
            ps2.append(corner2)
        # 各辺の点が2つ未満だと補完できない
        if len(ps1) < 2 or len(ps2) < 2:
            return None
        line1 = geo.Line.from_points_by_lms(ps1)
        line2 = geo.Line.from_points_by_lms(ps2)
        return line1.intersect(line2)
        
        
        


