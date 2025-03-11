import unittest
import numpy as np
from mark_seeker import MarkSeeker


class TestMarkSeeker(unittest.TestCase):

    def setUp(self):
        self.mark_seeker = MarkSeeker()

    def test_seek_mark_with_visible(self):
        buffer = np.zeros((1000, 1000, 3), dtype=np.uint8)
        obj_width = 20
        mark_length = 10
        ratio = 0.1
        visiable = True

        result, bottom = self.mark_seeker.seek_mark(
            buffer, obj_width, mark_length, ratio, visiable)
        self.assertFalse(result)
        self.assertEqual(bottom, -1)

    def test_seek_mark_with_non_visible(self):
        buffer = np.zeros((1000, 1000, 3), dtype=np.uint8)
        obj_width = 20
        mark_length = 10
        ratio = 0.1
        visiable = False

        result, bottom = self.mark_seeker.seek_mark(
            buffer, obj_width, mark_length, ratio, visiable)
        self.assertFalse(result)
        self.assertEqual(bottom, -1)


if __name__ == '__main__':
    unittest.main()
