import unittest
import numpy as np
from numpy import inf
import cv2
from mark_seeker import KongboSeeker


class TestKongboSeeker(unittest.TestCase):

    def setUp(self):
        self.seeker = KongboSeeker()

    def test_seek_mark_no_marks(self):
        buffer = np.zeros((100, 100, 3), dtype=np.uint8)
        obj_width = 10
        mark_length = 10
        result = self.seeker.seek_mark(buffer, obj_width, mark_length)
        self.assertEqual(result, (False, 0, [-inf]))
        self.assertEqual(result, (False, 0, [-np.inf]))


if __name__ == '__main__':
    unittest.main()
