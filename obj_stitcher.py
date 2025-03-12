import json
import cv2
import numpy as np
import os
from mark_seeker import KongboSeeker
from logging_config import logger

__DEBUG__ = False


class ObjStitcher:
    def __init__(self, mark_seeker, object_length_MM, cam_mm_per_pixel_Y, mark_length, frame_width_P, overlap=100):
        self.object_length_pixels = int(
            round(object_length_MM / cam_mm_per_pixel_Y))
        self.mark_length = mark_length
        self.overlap = overlap
        self.block_size = self.object_length_pixels + 2 * self.overlap
        self.frame_width_P = frame_width_P
        self.mark_roi = [0, self.object_length_pixels-self.mark_length+self.overlap,
                         self.frame_width_P, self.mark_length]

        self.buffer = []  # 存放连续帧
        self.mode = "seek_mark"  # 初始为seek mark模式

        self.global_mark_pos = None
        self.stitched_object = None

        self.mark_seeker = mark_seeker
        self.seek_mark: function = self.mark_seeker.seek_mark
        self.check_mark: function = self.mark_seeker.check_mark

    def process_frame(self, frame):

        completed_objects = []
        self.buffer.append(frame)
        total_rows = self._buffer_total_rows()
        logger.debug(f"当前缓冲区总行数: {total_rows}, 模式: {self.mode}")
        if self.mode == "seek_mark":
            mark_exists, mark_nums, mark_ends = self.seek_mark(
                self._buffer_to_array(), self.frame_width_P, self.mark_length, visiable=__DEBUG__)
            logger.debug(
                f"seek mark mode ：seek mark结果: {mark_exists}")
            if mark_exists:
                if total_rows >= mark_ends[-1] + self.overlap:
                    obj = self._extract_rows(0, mark_ends[-1]+self.overlap)
                    self._remove_rows(mark_ends[-1]-self.overlap)
                    completed_objects.append((obj, "seek_mark"))
                    self.mode = "check_mark"
                else:
                    # TODO 还需要考虑下一帧也有mark， 检测到下一个mark的情况
                    logger.debug(f"seek mark mode ：缓冲区不足，等待更多帧。")
        elif self.mode == "check_mark":

            if total_rows >= self.block_size:
                block = self._extract_rows(0, self.block_size)
                margin = self.overlap//2
                mark_exists, mark_end = self.check_mark(block, self.frame_width_P, self.mark_length,
                                                        self.mark_roi, margin, visiable=__DEBUG__)

                if mark_exists:
                    standard_mark_end = self.block_size-self.overlap
                    shift = mark_end - standard_mark_end
                    logger.debug(
                        f"mark end: {mark_end}, standard mark end: {standard_mark_end}")
                    logger.debug(f"偏移量: {shift}")
                    block = self._extract_rows(shift, self.block_size + shift)
                    completed_objects.append((block, "check_mark"))
                    self._remove_rows(self.block_size + shift - self.overlap*2)
                    logger.debug(f"输出 正常图，固定块: 0 ~ {self.block_size}")
                else:
                    self.problematic_pending = True
                    completed_objects.append((block, "problematic"))
                    logger.debug(f"固定块检测失败，定义为问题图:")
                    self.mode = "seek_mark"
            else:
                logger.debug(f"check mark mode ：缓冲区不足，等待更多帧。")
        return completed_objects

    def _buffer_to_array(self):
        """
        Concatenate all frames in the buffer to a single array.

        Returns:
            np.ndarray: A concatenated array of all frames in the buffer.
        """
        return np.concatenate(self.buffer, axis=0)

    def _buffer_total_rows(self):
        """
        Calculate the total number of rows in all frames stored in the buffer.

        This method iterates over each frame in the buffer and sums up the number
        of rows (shape[0]) for each frame.

        Returns:
            int: The total number of rows in all frames in the buffer.
        """
        return sum(frame.shape[0] for frame in self.buffer)

    def _extract_rows(self, start, end):
        """
        Extracts and concatenates rows from frames in the buffer within the specified range.

        Args:
            start (int): The starting row index (inclusive).
            end (int): The ending row index (exclusive).

        Returns:
            np.ndarray or None: A concatenated array of the extracted rows if any rows are within the range,
                                otherwise None.
        """
        total = 0
        segments = []
        for frame in self.buffer:
            rows = frame.shape[0]
            if total + rows <= start:
                total += rows
                continue
            s = max(0, start - total)
            e = min(rows, end - total)
            segments.append(frame[s:e])
            total += rows
            if total >= end:
                break
        return np.concatenate(segments, axis=0) if segments else None

    def _remove_rows(self, num_rows):
        rows_to_remove = num_rows
        new_buffer = []
        for frame in self.buffer:
            rows = frame.shape[0]
            if rows_to_remove >= rows:
                rows_to_remove -= rows
                continue
            new_buffer.append(frame[rows_to_remove:])
            rows_to_remove = 0
        self.buffer = new_buffer


def resize_image(image, ratio=0.01):
    return cv2.resize(image, (int(image.shape[1]*ratio), int(image.shape[0]*ratio)))


if __name__ == "__main__":
    file_path = r"D:\data\1"
    out_path = file_path+"_out"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    settings_file = os.path.join(file_path, "info.json")
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            json_content = json.load(f)
        object_length_MM = json_content["object_length_MM"]
        mark_length = json_content["mark_length"]
        cam_mm_per_pixel_Y = 1
        # frame_width_P = 8192
        frame_width_P = 7200  # 测试图有点歪，我为了方便用这个截了下右边部分
        overlap = 500
    else:
        raise FileNotFoundError("settings file 'info.json' not found!")

    mark_seeker = KongboSeeker()
    obj_stitcher = ObjStitcher(
        mark_seeker, object_length_MM, cam_mm_per_pixel_Y, mark_length, frame_width_P, overlap)

    from frame_capturer import FrameCapturer
    frame_capture = FrameCapturer(mode='local', file_path=file_path)
    while frame_capture.ready_to_capture():
        frame = frame_capture.next_frame()
        frame_name = frame_capture.current_file_name
        frame = frame[:, :frame_width_P]
        completed_objects = obj_stitcher.process_frame(frame)
        for obj, obj_type in completed_objects:
            logger.debug(f"检测到类型：{obj_type}")
            if not os.path.exists(os.path.join(out_path, obj_type)):
                os.makedirs(os.path.join(out_path, obj_type))
            cv2.imwrite(os.path.join(out_path, obj_type,
                        frame_name), resize_image(obj))
