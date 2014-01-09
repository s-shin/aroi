# -*- coding: utf-8 -*-
"""マーカー解析のためのモジュール。

.. code-block:: python

    src_img = cv2.imread(sys.argv[1])

    # 前処理（2値化）
    bin_img, _ = preprocess_image(src_img)
    
    # 四角形の解析。
    # squaresは4つの頂点（配列）の集合（配列）のリスト（配列）
    squares = detect_squares(bin_img)

    if not len(squares):
        print "no square detected."
        return

    # マーカー候補画像の取得
    marker_imgs = [extract_image(bin_img, s) for s in squares]
    
    # マーカー判定
    detectors = [detect_black_marker]
    for img in marker_imgs:
        result = detect_marker(img, detectors)
        if result is None:
            continue
        # ...
"""
import copy
import math
import numpy as np
import cv2

def intersects(a, b, c, d):
    """頂点a,b,c,dについて、線分abが線分cdと交差しているか調べる。
    
    :param a,b,c,d: 頂点を表す2次元配列
    :type a,b,c,d: list
    :returns: 交差していたらTrue
    :rtype: bool
    
    >>> marker.intersects([0, 0], [1, 1], [1, 0], [1, 1])
    True
    >>> marker.intersects([0, 0], [1, 0], [1, 1], [0, 1])
    False
    
    .. note::
       :func:`.detect_squares` の内部で利用される。通常、ユーザが使う必要は無い。

    .. seealso::
       `交差判定のアルゴリズム <http://www5d.biglobe.ne.jp/~tomoya03/shtml/algorithm/Intersection.htm>`_
    """
    tc = (a[0]-b[0]) * (c[1]-a[1]) + (a[1]-b[1]) * (a[0]-c[0])
    td = (a[0]-b[0]) * (d[1]-a[1]) + (a[1]-b[1]) * (a[0]-d[0])
    # 稀に算術オーバーフローするので
    tc = 1 if tc > 0 else -1
    td = 1 if td > 0 else -1
    return tc * td < 0

def are_in_clockwise_order(points):
    """閉区間を形成する頂点列が要素順に辿った時に時計回りかどうかを判定する。
    座標系は下方向y軸正、右方向x軸正。
    
    :param points: 頂点を表す2次元配列の配列。これらの頂点は配列の要素順に線を結んだ時に閉区間を形成しなければならない。
    :type points: list
    :returns: 時計回りならTrue
    :rtype: bool
    
    >>> marker.are_in_clockwise_order([[0, 0], [1, 0], [1, 1], [0, 1]])
    True
    >>> marker.are_in_clockwise_order([[0, 0], [0, 1], [1, 1], [1, 0]])
    False
    
    .. note::
       :func:`.detect_squares` の内部で利用される。通常、ユーザが使う必要は無い。
    
    .. seealso::
       `回転方向判定のアルゴリズム <http://www5d.biglobe.ne.jp/~noocyte/Programming/Geometry/PolygonMoment-jp.html>`_
    """
    s = 0
    for i in range(len(points)):
        j = i + 1 if i != len(points)-1 else 0
        s += (points[i][0] + points[j][0]) * (points[i][1] - points[j][1])
    return s < 0

def preprocess_image(src_img, blur=(5, 5)):
    """画像の前処理。
    :func:`.detect_squares` の前に利用。2値化された画像を取得できる。
    
    :param src_img: 生の入力画像
    :returns: 2値化された画像, グレースケールの画像
    """
    gs_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    if blur:
        gs_img = cv2.GaussianBlur(gs_img, blur, 0)
    r, bin_img = cv2.threshold(
        gs_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return bin_img, gs_img

def detect_squares(bin_img, min_size=400, param_n=4, param_a=45):
    """画像から四角形を検出する。
    事前に :func:`.preprocess_image` （もしくは自力）で画像を2値化しておくこと。
    
    :param bin_img: 2値化済みの入力画像。
    :type bin_img: cv2 image
    :param min_size: 単位はpx。このpx数以上の領域のみ抽出される。
    :type min_size: int
    :param param_n: 対の辺の長さが ``param_n`` 倍以上のものを排除。
    :param param_a: 対の辺のなす角が ``param_a`` 度以上のものを排除。対の辺を似た方向であるベクトルで表現し、それらのなす角を調べる。
    :returns: 近似枠情報（4頂点、時計回り）の配列
    :rtype: list
    
    .. note::
        戻り値の形式はPythonのリストとnumpyの配列が混ざっているので注意。
        
        >>> print type(squares), type(squares[0]), type(squares[0][0])
        <type 'list'> <type 'numpy.ndarray'> <type 'numpy.ndarray'>
    """
    tmp_img = copy.deepcopy(bin_img)
    contours, hierarchy = cv2.findContours(
        tmp_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    for contour in contours:
        # min_size[px]未満のものを排除
        if cv2.contourArea(contour) < min_size:
            continue
        # 擬似ポリゴンの頂点を取得。
        corners = cv2.approxPolyDP(contour, 3, True)
        # 頂点数が4ではないものを排除
        if len(corners) != 4:
            continue

        # 以下の方法で明らかに形状が歪なものを排除する。
        # [1] なす角が90度を越す形。（3点の中に1点が含まれる形）
        # [2] 対となる辺が`param_n`倍以上長さが違う。
        # [3] 対となる辺のベクトルのなす角が`param_a`度以上になっている。

        # 抽出すべき形状は、適当な2点p・qについて、
        # 線分prとqs、psとqrのどちらかが交差しているはずである。
        # これにより[1]を排除できる。
        # OpenCVの仕様上、p, q, r, sを順にたどると閉区間を形成するので、
        # 一回の判定で良い。
        [[p], [q], [r], [s]] = corners # 配列のネストが一つ多い
        if not intersects(p, r, q, s):
            continue
        va1 = q - p
        va2 = r - s # va1と似た向き
        vb1 = r - q
        vb2 = s - p # vb1と似た向き
        # [2]をチェックする。長さを比較するだけ。
        [va1d, va2d, vb1d, vb2d] = map(np.linalg.norm, [va1, va2, vb1, vb2])
        if max(va1d, va2d) / min(va1d, va2d) > param_n:
            continue
        if max(vb1d, vb2d) / min(vb1d, vb2d) > param_n:
            continue
        # [3]をチェックする。内積から角度を逆算。
        rad_param_a = param_a * math.pi / 180 # 速度的に必要ならループ外に出す
        va1va2 = va1 * va2 # numpyによる要素ごとの掛け算
        # 稀に丸め誤差で1を超すことがあるようなのでminしておく
        t = min(1.0, (va1va2[0]+va1va2[1]) / (va1d*va2d))
        va1va2_angle = math.acos(t)
        if va1va2_angle > rad_param_a:
            continue
        vb1vb2 = vb1 * vb2 # numpyによる要素ごとの掛け算
        t = min(1.0, (vb1vb2[0]+vb1vb2[1]) / (vb1d*vb2d))
        vb1vb2_angle = math.acos(t)
        if vb1vb2_angle > rad_param_a:
            continue
        # 最後に頂点列を時計回りにする。
        # これによりOpenCVでの透視変換の際に画像が反転することが無くなる。
        if are_in_clockwise_order([p, q, r, s]):
            results.append(np.array([p, q, r, s]))
        else:
            results.append(np.array([p, s, r, q]))
        
    return results

def extract_image(src_img, square_corners, size=(100, 100)):
    """``src_img`` から ``square_corners`` の領域をくり抜き、正方形に修正する。
    
    :param src_img: 入力画像
    :param square_corners: 4つの頂点。通常 :func:`.detect_squares()` で検出したもの。
    :param size: 単位はpx。変換後の正方形のサイズ。
    :type size: (int, int)
    """
    src = np.array(square_corners, np.float32)
    w, h = size
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(src_img, matrix, (w, h))

def detect_marker(marker_img, detectors):
    """マーカー検出のためのインターフェース
    
    この関数を利用してマーカー検出を行う場合は、検出器(detector)は
    次のインターフェースを満たさなければならない。
    
    - 第1引数に ``marker_img`` を受け取る。
    - 検出できなかった場合はNoneを返す。検出できた場合は、任意のデータ構造で
      検出結果に関する情報を返す。
    
    :param marker_img: マーカー画像
    :param detectors: 検出器の配列。
    :returns: 検出器の要素番号と検出結果情報。検出できなかった場合は要素番号部分が-1になる。
    :rtype: (int, any type)
    
    .. note::
       :func:`.detect_black_marker` 、 :func:`.detect_white_marker` 、
       :func:`.detect_white_square_in_black_marker` は検出器の実装例。
       
       検出器の入力画像形式が異なる場合は、一回の :func:`detect_marker` で
       全ての判定を行うことが出来ない点に注意。
    """
    for i in range(len(detectors)):
        result = detectors[i](marker_img)
        if result is not None:
            return i, result
    return -1, None

def detect_black_marker(bin_img, threshold=0.9):
    """大部分が黒いマーカーの検出。
    全体の ``threshold`` 割が黒だったらマーカーとみなされる。
    回転は判定できない。
    ``threshold`` の値を変えたい時は ``functools.partial()`` を使うこと。
    
    :param bin_img: 2値化されているマーカー画像。
    :param threshold: しきい値。
    :type threshold: float
    """
    black_num = len(bin_img[bin_img==0])
    if black_num > bin_img.size * threshold:
        return True
    return None

def detect_white_marker(bin_img, threshold=0.9):
    """:func:`.detect_black_marker` の逆の処理。
    """
    white_num = len(bin_img[bin_img>0])
    if white_num > bin_img.size * threshold:
        return True
    return None

def detect_white_square_in_black_marker(bin_img, threshold=0.7):
    """白色の正方形が黒の中に含まれているマーカーの検出。
    マーカーの1辺の半分の長さの辺を持つ白の正方形が中央にあるかチェックする。
    
    :param img: マーカー画像。
    :param threshold: しきい値。
    """
    height, width = bin_img.shape
    qh = height * 0.25
    qw = width * 0.25
    # 白のROI
    white_img = bin_img[qh:height-qh, qw:width-qw]
    is_white = detect_white_marker(white_img, threshold)
    if not is_white:
        return None
    # 黒のROI
    black_imgs = [
        bin_img[0:qh,:],
        bin_img[height-qh:height,:],
        bin_img[qh:height-qh,0:qw],
        bin_img[qh:height-qh,width-qw:width]
    ]
    black_num = 0
    size = 0
    for img in black_imgs:
        black_num += len(img[img==0])
        size += img.size
    is_black = black_num > size * threshold
    if not is_black:
        return None
    return True


class SimpleMarkerDetector(object):
    """非常にシンプルなマーカーの定義（主にテスト用）。
    
    検出器の実装方法の一例でもある。このようにマーカーをクラスで定義すれば、
    :func:`detect_marker` のデータ構造に一貫性を持たせることが出来る。
    
    .. code-block:: python
    
        detectors = [
            detect_black_marker,
            SimpleMarker().detect
        ]
    
    """
    
    BLACK_BIT = 1
    WHITE_BIT = 0
    
    IMAGE = np.array([
        [0, 0, 0, 0],
        [0, 0, 255, 0],
        [0, 255, 255, 0],
        [0, 0, 0, 0],
    ])
    
    @classmethod
    def save_marker_image(cls, filename="marker.png", size=512):
        """このメソッドでマーカー画像をファイルに保存できる。
        """
        cv2.imwrite(filename, cv2.resize(cls.IMAGE, (size, size),
            interpolation=cv2.INTER_NEAREST))
    
    def __init__(self, marker_frame_relative_size=0.25, threshold=0.9):
        """
        :param marker_frame_relative_size: マーカーの黒縁の幅のマーカー1辺に対する割合。
        :param threshold: ビット判定時のしきい値。大きいほど厳密。
        """
        self.threshold = threshold
        self.marker_frame_relative_size = marker_frame_relative_size

    def img2bit(self, img):
        """画像の白/黒の割合から0/1ビットに変換する。
        """
        size = img.size
        black_pixel_num = len(img[img==0])
        white_pixel_num = size - black_pixel_num
        if black_pixel_num > size * self.threshold:
            return self.BLACK_BIT
        elif white_pixel_num > size * self.threshold:
            return self.WHITE_BIT
        else:
            return None

    def detect(self, bin_img):
        """
        :returns: 検出できた場合は、正しい向きに対して何回右90度回転しているかを返す。
        :rtype rot90: int/None
        """
        # TODO: 黒縁チェックを入れるか。
        img_size = len(bin_img)
        frame_size = img_size * self.marker_frame_relative_size
        pattern_region = bin_img[
            frame_size : img_size - frame_size,
            frame_size : img_size - frame_size
        ]
        bit_size = (img_size - frame_size * 2) * 0.5
        tl = self.img2bit(pattern_region[:bit_size,:bit_size])
        tr = self.img2bit(pattern_region[:bit_size,bit_size:])
        bl = self.img2bit(pattern_region[bit_size:,:bit_size])
        br = self.img2bit(pattern_region[bit_size:,bit_size:])
        bits = [tl, tr, br, bl]
        if None in bits:
            return None
        if len(filter(lambda x: x == self.BLACK_BIT, bits)) != 1:
            return None
        rot90 = bits.index(self.BLACK_BIT)
        return rot90


class TemplateDetector(object):
    """テンプレートマッチングを利用したマーカー検出。
    """
    
    def __init__(self, size=(128, 128), threshold=0.9):
        self.size = size
        self.threshold = threshold
        self.templates = {}
    
    def prepare(self):
        """templatesプロパティを書き換えたらこのメソッドを呼び出す。
        """
        self._templates = {}
        for name, template_img in self.templates.items():
            tmp = cv2.resize(template_img, self.size)
            tmp, _ = preprocess_image(tmp)
            self._templates[name] = [np.rot90(tmp, i) for i in range(4)]
    
    def detect(self, bin_img):
        """検出メソッド
        
        :param bin_img: 2値化済みマーカー画像
        :returns: 検出できなかったらNone。検出できた場合は、マーカー名と90度回転数を返す。
        """
        target_img = cv2.resize(bin_img, self.size)
        for name, template in self._templates.items():
            for i, template_img in enumerate(template):
                result = cv2.matchTemplate(
                    target_img, template_img, cv2.TM_CCOEFF_NORMED)
                if result > self.threshold:
                    return name, i
        return None
    
