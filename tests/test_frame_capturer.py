import shutil
import tempfile
import unittest
import os

import cv2
from frame_capturer import FrameCapturer


class TestFrameCapturer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("Setting up for test")
        cls.source_image_dir = r'./data/1'
        cls.test_image_dir = 'test_images'
        os.makedirs(cls.test_image_dir, exist_ok=True)
        # copy first three images from source_image_dir to test_image_dir with new names
        cls.test_images = []
        for i, img_file in enumerate(os.listdir(cls.source_image_dir)[:3]):
            img = cv2.imread(os.path.join(cls.source_image_dir, img_file))
            cv2.imwrite(os.path.join(cls.test_image_dir, f'{img_file}'), img)
            cls.test_images.append(f'{img_file}')

    @classmethod
    def tearDownClass(cls):
        print("clearing up after test")
        shutil.rmtree(cls.test_image_dir)

    def test_init_local_mode(self):
        capturer = FrameCapturer(mode='local', file_path=self.test_image_dir)
        self.assertEqual(capturer.file_path, self.test_image_dir)
        self.assertEqual(capturer.file_list, self.test_images)
        self.assertEqual(capturer.current_file_idx, -1)
        self.assertEqual(capturer.total_files, len(self.test_images))

    def test_init_camera_mode(self):
        with self.assertRaises(NotImplementedError):
            FrameCapturer(mode='camera')

    def test_init_invalid_mode(self):
        with self.assertRaises(ValueError):
            FrameCapturer(mode='invalid')

    def test_init_local_mode_no_file_path(self):
        with self.assertRaises(ValueError):
            FrameCapturer(mode='local')

    def test_init_local_mode_nonexistent_path(self):
        with self.assertRaises(FileNotFoundError):
            FrameCapturer(mode='local', file_path='nonexistent_path')

    def test_ready_to_capture(self):
        capturer = FrameCapturer(mode='local', file_path=self.test_image_dir)
        self.assertTrue(capturer.ready_to_capture())
        capturer.current_file_idx = len(self.test_images)
        self.assertFalse(capturer.ready_to_capture())

    def test_next_frame(self):
        capturer = FrameCapturer(mode='local', file_path=self.test_image_dir)
        for i in range(len(self.test_images)):
            frame = capturer.next_frame()
            self.assertIsNotNone(frame)
        with self.assertRaises(IndexError):
            capturer.next_frame()

    def test_next_frame_with_no_images(self):
        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaises(FileNotFoundError):
                capturer = FrameCapturer(mode='local', file_path=empty_dir)
                capturer.next_frame()


if __name__ == '__main__':
    unittest.main()
