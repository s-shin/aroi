# -*- coding: utf-8 -*-
"""シンプルな4隅マーカー検出モジュール。
3つの黒いマーカーと、1つの黒領域の中心に白領域のあるマーカー、計4つのマーカーを1セットとするROI型ARの実現が目的。
"""
import copy
import cv2
import numpy as np
from ... import marker


class Detector(object):
    
    def detect(self, src_img, draw=False):
        # 四角形の検出
        bin_img, _ = marker.preprocess_image(src_img)
        squares = marker.detect_squares(bin_img, 15**2)
        imgs = [marker.extract_image(bin_img, s) for s in squares]
        # マーカーの検出
        detectors = [
            marker.detect_black_marker,
            marker.detect_white_square_in_black_marker
        ]
        markers = []
        for i in range(len(imgs)):
            img = imgs[i]
            idx, result = marker.detect_marker(img, detectors)
            if idx < 0:
                continue
            markers.append({
                "detector": idx,
                "result": result,
                "img": img,
                "square": squares[i]
            })
        # とりあえず4つ以上検出した時は無視する。
        if len(markers) != 4:
            return None
        # 0のマーカーが3つ、1のマーカーが1つであれば続行。
        if reduce(lambda acc, m: acc + (1 if m["detector"] == 0 else 0),
                markers, 0) != 3:
            return None
        # 中央点を取得。
        sp = [np.average(m["square"], axis=0) for m in markers]
        # 1のマーカーを配列の先頭に持ってくる。
        # 以降、先頭の要素は固定。
        for i in range(len(markers)):
            if markers[i]["detector"] == 1:
                break
        sp = sp[i:] + sp[:i] # rotation
        # 要素順に繋げた時に交差しないよう並べ替える。
        if marker.intersects(*sp):
            sp[1], sp[2] = sp[2], sp[1]
        else:
            if marker.intersects(sp[0], sp[3], sp[1], sp[2]):
                sp[2], sp[3] = sp[3], sp[2]
        # 要素順に繋げた時に右回りになるようにする。
        if not marker.are_in_clockwise_order(sp):
            sp[1], sp[3] = sp[3], sp[1]
        # デバッグ用描画
        if draw:
            t = np.int32(sp)
            cv2.circle(src_img, tuple(t[0]), 1, (0, 255, 0), 3)
            cv2.polylines(src_img, [t], True, (0, 255, 0))
        return sp
    
    def get_points_for_calibration(self, square_points, square_size=1):
        """キャリブレーションのための点列を返す。精度が悪い。
        
        :param square_points: 4 points representing square in clockwise order.
        :param square_size: Square width(=height). An object whose size is ``square_size`` norm in OpenGL coordinates is drawed as real square size in the picture.
        """
        sp = np.float32(square_points)
        t = copy.deepcopy(sp[1]); sp[1] = sp[2]; sp[2] = t
        obj_points = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        obj_points *= square_size
        return sp, obj_points
    
    def get_points_for_estimation(self, square_points, square_size=1):
        """位置推定のための点列を返す。
        
        :param square_points: 4 points representing square in clockwise order.
        :param square_size: Square width(=height). An object whose size is ``square_size`` norm in OpenGL coordinates is drawed as real square size in the picture.
        """
        obj_points = np.float32(
            [[0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0]]) - [0.5, 0, 0.5]
        obj_points *= square_size
        return np.float32(square_points), obj_points
        

