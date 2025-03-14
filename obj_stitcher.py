import json
import cv2
import numpy as np
import os
from mark_seeker import KongboSeeker
from logging_config import logger


__DEBUG__ = False


class ObjStitcher:
    """
    ObjStitcher is a class designed to process frames of an object and stitch them together based on detected marks.
    Attributes:
        object_length_pixels (int): The length of the object in pixels.
        input_type (str): The type of input, either 'aligned' or 'not_aligned'.
        mark_to_end (int): The distance from the mark to the end of the object.
        overlap (int): The overlap in pixels.
        block_size (int): The size of the block in pixels.
        mark_roi (list): The region of interest for the mark.
        buffer (list): A list to store consecutive frames.
        mode (str): The current mode, either "seek_mark" or "check_mark".
        seek_mark (function): A function to seek marks.
        check_mark (function): A function to check marks.
        frame_id (int): The current frame ID.
        frame_list (list): A list of frame IDs.
    Methods:
        process_frame(frame, reset_signal=False):
            Processes a single frame and updates the buffer and mode accordingly.
                frame (np.ndarray): The frame to process.
                reset_signal (bool, optional): If True, resets the object. Defaults to False.
                dict: A dictionary containing the completed object and its type.
        _buffer_to_array():
            Concatenates all frames in the buffer to a single array.
        _buffer_total_rows():
            Calculates the total number of rows in all frames stored in the buffer.
        _extract_rows(start, end):
                np.ndarray or None: A concatenated array of the extracted rows if any rows are within the range, otherwise None.
        _remove_rows(num_rows):
    """

    def __init__(self, mark_seeker, object_length_MM, cam_MM_per_P_Y, mark_length, frame_width_P, input_type='aligned', overlap=100):
        """
        Initializes the ObjStitcher class with the given parameters.
        Args:
            mark_seeker (object): An object responsible for seeking and checking marks.
            object_length_MM (float): The length of the object in millimeters.
            cam_MM_per_P_Y (float): The camera's millimeters per pixel in the Y direction.
            mark_length (int): The length of the mark in pixels.
            frame_width_P (int): The width of the frame in pixels.
            input_type (str, optional): The type of input, either 'aligned' or 'not_aligned'. Defaults to 'aligned'.
            overlap (int, optional): The overlap in pixels. Defaults to 100.
        Raises:
            ValueError: If the input_type is not 'aligned' or 'not_aligned'.
        """

        self.object_length_pixels = int(
            round(object_length_MM / cam_MM_per_P_Y))

        self.mark_length = mark_length
        self.input_type = input_type
        if self.input_type == "aligned":
            self.mark_to_end = 0
        elif self.input_type == "not_aligned":
            self.mark_to_end = self.object_length_pixels - self.mark_length
        else:
            raise ValueError("input type must be 'aligned' or 'not_aligned'")

        self.overlap = overlap
        self.block_size = self.object_length_pixels + 2 * self.overlap
        self.frame_width_P = frame_width_P
        self.mark_roi = [0, self.object_length_pixels - self.mark_length + self.overlap - self.mark_to_end,
                         self.frame_width_P, self.mark_length]

        self.buffer = []  # 存放连续帧
        self.mode = "seek_mark"  # 初始为seek mark模式

        self.mark_seeker = mark_seeker
        self.seek_mark: function = self.mark_seeker.seek_mark
        self.check_mark: function = self.mark_seeker.check_mark
        self.frame_id = 0
        self.frame_list = []

    def process_frame(self, frame, reset_signal=False):
        """
        Process a single frame and manage the internal buffer and state.
        Args:
            frame (numpy.ndarray): The frame to be processed.
            reset_signal (bool, optional): If True, resets the internal state by reinitializing the object. Defaults to False.
        Returns:
            dict: A dictionary containing the completed object and its metadata. The dictionary has the following keys:
                - "object" (numpy.ndarray): The processed object or block.
                - "type" (str): The type of processing result, either "seek_mark_success", "check_mark_success", or "check_mark_fail".
                - "mark_ends" (list or int): The end positions of the marks found, or -np.inf if mark check failed.
                - "frame_list" (list, optional): List of frame IDs, included only for "seek_mark_success".
        """
        if reset_signal:
            self.__init__()
        self.frame_id += 1
        self.frame_list.append(self.frame_id)
        completed_objects = dict()
        self.buffer.append(frame)
        total_rows = self._buffer_total_rows()
        logger.debug(f"[READ FRAME] 当前缓冲区总行数: {total_rows}, 模式: {self.mode}")
        if self.mode == "seek_mark":
            mark_exists, mark_nums, mark_ends = self.seek_mark(
                self._buffer_to_array(), self.frame_width_P, self.mark_length, visiable=__DEBUG__)
            logger.debug(
                f"[SEEK MARK] result: {mark_exists}")
            if mark_exists:
                if total_rows >= mark_ends[-1] + self.mark_to_end + self.overlap:
                    obj = self._extract_rows(
                        0, mark_ends[-1]+self.overlap + self.mark_to_end)
                    self._remove_rows(
                        mark_ends[-1]-self.overlap + self.mark_to_end)
                    completed_objects = {
                        "object": obj,
                        "type": "seek_mark_success",
                        "mark_ends": mark_ends,
                        "frame_list": []
                    }
                    self.mode = "check_mark"
                else:
                    # TODO 还需要考虑下一帧也有mark， 检测到下一个mark的情况
                    logger.debug(f"[SEEK MARK] 缓冲区不足，等待更多帧。")
        elif self.mode == "check_mark":

            if total_rows >= self.block_size:
                block = self._extract_rows(0, self.block_size)
                margin = self.overlap//2
                mark_exists, mark_end = self.check_mark(block, self.frame_width_P, self.mark_length,
                                                        self.mark_roi, margin, visiable=__DEBUG__)

                if mark_exists:
                    standard_mark_end = self.block_size-self.overlap - self.mark_to_end
                    shift = mark_end - standard_mark_end
                    logger.debug(
                        f"[CHECK MARK]: mark end: {mark_end}, standard mark end: {standard_mark_end}")
                    logger.debug(f"[CHECK MARK] 偏移量: {shift}")
                    block = self._extract_rows(shift, self.block_size + shift)
                    completed_objects = {
                        "object": block,
                        "type": "check_mark_success",
                        "mark_ends": mark_end
                    }
                    self._remove_rows(self.block_size + shift - self.overlap*2)
                    logger.debug(
                        f"[CHECK MARK]: 输出 正常图，固定块: 0 ~ {self.block_size}")
                else:
                    self.problematic_pending = True
                    completed_objects = {
                        "object": block,
                        "type": "check_mark_fail",
                        "mark_ends": -np.inf
                    }
                    logger.debug(f"[CHECK MARK]: 固定块检测失败，定义为问题图:")
                    self.mode = "seek_mark"
            else:
                logger.debug(f"[CHECK MARK] ：缓冲区不足，等待更多帧。")
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
        """
        Removes a specified number of rows from the beginning of each frame in the buffer.

        Args:
            num_rows (int): The number of rows to remove from the buffer.

        The method iterates through each frame in the buffer and removes the specified number of rows.
        If the number of rows to remove is greater than or equal to the number of rows in a frame,
        the entire frame is skipped. The remaining frames are adjusted accordingly and stored back
        in the buffer.
        """
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
    """
    Resize the given image by a specified ratio.

    Parameters:
    image (numpy.ndarray): The input image to be resized.
    ratio (float, optional): The ratio by which to resize the image. Default is 0.01.

    Returns:
    numpy.ndarray: The resized image.
    """
    return cv2.resize(image, (int(image.shape[1]*ratio), int(image.shape[0]*ratio)))


if __name__ == "__main__":
    test_path_list = ['1', '2', '3', '5', '6', '7', '8', '9', '10']
    for t in test_path_list:
        file_path = os.path.join('D:\data', t)
        settings_file = os.path.join(file_path, "info.json")
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                json_content = json.load(f)
            object_length_MM = json_content["object_length_MM"]
            mark_length = json_content["mark_length"]
            cam_MM_per_P_Y = 1
            # frame_width_P = 8192
            frame_width_P = 7200  # 测试图有点歪，我为了方便用这个截了下右边部分
            overlap = 200
            if overlap > mark_length//3:
                overlap = mark_length//3
            mark_seeker = KongboSeeker()
            obj_stitcher = ObjStitcher(
                mark_seeker, object_length_MM, cam_MM_per_P_Y, mark_length, frame_width_P, "not_aligned", overlap)

        else:
            raise FileNotFoundError("settings file 'info.json' not found!")

        out_path = file_path + "_out" + "_" + obj_stitcher.input_type
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        from frame_capturer import FrameCapturer
        frame_capture = FrameCapturer(mode='local', file_path=file_path)
        while frame_capture.ready_to_capture():
            frame = frame_capture.next_frame()
            frame_name = frame_capture.current_file_name
            frame = frame[:, :frame_width_P]
            completed_objects = obj_stitcher.process_frame(frame)
            if completed_objects:
                obj = completed_objects["object"]
                obj_type = completed_objects["type"]
                mark_ends = completed_objects["mark_ends"]
                if "fail" in obj_type:
                    logger.error(f"---------RESULT---------{obj_type}")
                else:
                    logger.debug(f"---------RESULT---------{obj_type}")
                if not os.path.exists(os.path.join(out_path, obj_type)):
                    os.makedirs(os.path.join(out_path, obj_type))
                cv2.imwrite(os.path.join(out_path, obj_type,
                            frame_name), resize_image(obj, 0.01))
