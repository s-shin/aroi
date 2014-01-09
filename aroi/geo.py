# -*- coding: utf-8 -*-
"""幾何学系の処理の実装。

* 回帰直線
* 交点
"""

class Line(object):
    """数式 ``ax + by + c = 0`` を表す直線クラス。
    """
    
    def __init__(self, a, b, c):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
    
    def __str__(self):
        return "{0} * x + {1} * y + {2} = 0".format(self.a, self.b, self.c)

    # @classmethod
    # def from_points_by_rma(cls, points):
    #     """RMA（Reduced Major Axis）による直線の生成。
    # 
    #     .. seealso::
    #         http://vege1.kan.ynu.ac.jp/nakaumi/turnover/regression.html
    #         http://www7.atwiki.jp/hayatoiijima/pages/23.html
    #     """
    #     raise NotImplementedError
    
    @classmethod
    def from_points_by_lms(cls, points):
        """最小二乗法（least mean square）による直線の生成。
        ``points[n] = (x[n], y[n])`` について最もフィットする直線を生成。
        
        .. seealso::
            http://szksrv.isc.chubu.ac.jp/lms/lms1.html
        """
        n = len(points)
        sigma_x = sigma_y = sigma_xx = sigma_xy = 0
        for p in points:
            sigma_x += p[0]
            sigma_y += p[1]
            sigma_xx += p[0] * p[0]
            sigma_xy += p[0] * p[1]
        a = n * sigma_xy - sigma_x * sigma_y
        b = sigma_x**2 - n * sigma_xx
        c = sigma_xx * sigma_y - sigma_xy * sigma_x
        return cls(a, b, c)
    
    def intersect(self, line):
        """2つの線分の交点を返す。
        
        :return: 交差していたら交点を、していなかったらNoneを返す。
        """
        det = self.a * line.b - self.b * line.a
        if det == 0:
            return None
        x = (self.b * line.c - line.b * self.c) / det
        y = (line.a * self.c - self.a * line.c) / det
        return x, y

