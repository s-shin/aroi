# -*- coding: utf-8 -*-
import unittest
import argparse
from aroi import utils


class UtilsModuleTest(unittest.TestCase):
    
    def test_capture_setup_argparser(self):
        parser = utils.Capture.setup_argparser(utils.ThrowingArgumentParser())
        args = parser.parse_args("-v test.mpg".split())
        self.assertEqual(args.video, "test.mpg")
        args = parser.parse_args("-c 1".split())
        self.assertEqual(args.camera, 1)
        args = parser.parse_args("-c".split())
        self.assertEqual(args.camera, 0)
        with self.assertRaises(utils.ArgumentParserError):
            parser.parse_args("".split())
        with self.assertRaises(utils.ArgumentParserError):
            parser.parse_args("-v".split())


if __name__ == "__main__":
    unittest.main()



