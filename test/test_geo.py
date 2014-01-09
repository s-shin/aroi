# -*- coding: utf-8 -*-
import unittest
from aroi import geo

class GeoModuleTest(unittest.TestCase):
    
    def test_lms(self):
        p = geo.Line.from_points_by_lms([(0, 0), (1, 1)])
        self.assertEqual(p.a/(-p.b), 1) # 傾き
        self.assertEqual(p.c/(-p.b), 0) # y切片
        
        p = geo.Line.from_points_by_lms([(0, 0), (1, 1), (2, 3), (3, 5)])
        self.assertGreater(p.a/(-p.b), 1) # 傾き
        self.assertLess(p.c/(-p.b), 0) # y切片
        
        p = geo.Line.from_points_by_lms([(0, 0), (1, 0)]) # y = 0
        self.assertEqual(p.a, 0)
        self.assertNotEqual(p.b, 0)
        self.assertEqual(p.c, 0)
        
        """これがパスしない…
        p = geo.Line.from_points_by_lms([(0, 0), (0, 1)]) # x = 0
        self.assertNotEqual(p.a, 0)
        self.assertEqual(p.b, 0)
        self.assertEqual(p.c, 0)
        """
    
    def test_intersection(self):
        ps1 = [(0, 0), (1, 1)]
        ps2 = [(0, 1), (1, 0)]
        line1 = geo.Line.from_points_by_lms(ps1)
        line2 = geo.Line.from_points_by_lms(ps2)
        p = line1.intersect(line2)
        self.assertEqual(p, (0.5, 0.5))


if __name__ == "__main__":
    unittest.main()







