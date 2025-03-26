import copy
import time
import json
import traceback
import cv2
import numpy as np
import os
from mark_seeker import KongboSeeker
from logging_config import logger
import pdb

__DEBUG__ = True
CAM_MUL_FACTOR = 1
CAM_DIV_FACTOR = 1
CAM_PULSE_PER_PIXEL = CAM_DIV_FACTOR / CAM_MUL_FACTOR


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

    def __init__(self):
        logger.info(
            "to work with timesai, instance has to be initialised manually")
        self.initialised = False

    def _init_timesai_tool(self, mark_seeker, jp_length_MM, mark_length_MM,
                           cam_MM_per_P_Y, frame_width_P, input_type,
                           overlap_P):
        """
        Initializes the ObjStitcher class with the given parameters.
        Args:
            mark_seeker (object): An object responsible for seeking and checking marks.
            object_length_MM (float): The length of the object in millimeters.
            cam_MM_per_P_Y (float): The camera's millimeters per pixel in the Y direction.
            mark_length_MM (int): The length of the mark in pixels.
            frame_width_P (int): The width of the frame in pixels.
            input_type (str, optional): The type of input, either 'aligned' or 'not_aligned'. Defaults to 'aligned'.
            overlap (int, optional): The overlap in pixels. Defaults to 100.
        Raises:
            ValueError: If the input_type is not 'aligned' or 'not_aligned'.
        """
        self.cam_MM_per_P_Y = cam_MM_per_P_Y
        self.obj_length_MM = jp_length_MM + mark_length_MM
        self.object_length_P = int(round(self.obj_length_MM / cam_MM_per_P_Y))
        self.mark_length_P = int(round(mark_length_MM / cam_MM_per_P_Y))
        self.jp_length_P = self.object_length_P - self.mark_length_P
        print(self.object_length_P,
              self.mark_length_P,
              jp_length_MM,
              cam_MM_per_P_Y,
              mark_length_MM,
              frame_width_P,
              flush=True)
        self.input_type = input_type
        if self.input_type == "aligned":
            """
            |-------|
            |       |   
            |       |
            |       |
            |       |   
            |*******|
            |*******|
            |*******|
            """
            self.mark_to_end = 0
        elif self.input_type == "not_aligned":
            """
            |*******|
            |*******|   mark_length
            |*******|
            |       |
            |       |   jp_length
            |       |
            |       |
            |-------|
            """
            self.mark_to_end = self.jp_length_P
        else:
            raise ValueError("input type must be 'aligned' or 'not_aligned'")

        self.overlap_P = overlap_P
        self.block_size = self.object_length_P + 2 * self.overlap_P
        self.frame_width_P = frame_width_P
        self.mark_roi = [
            0, self.jp_length_P + self.overlap_P - self.mark_to_end,
            self.frame_width_P, self.mark_length_P
        ]

        # self.buffer = []
        self.buffer = None  # 存放连续帧

        self.mark_seeker = mark_seeker
        self.seek_mark: function = self.mark_seeker.seek_mark
        self.check_mark: function = self.mark_seeker.check_mark
        self.mode = "seek_mark"  # 初始为seek mark模式

        self.frame_id = 0
        self.frame_list = []
        self.frame_offsets = []
        self.current_frame_in_obj_start = 0
        self.first_round = True

        self.initialised = True  # 初始化成功

    def process_frame(self, frame, frame_pulse):
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
        completed_objects = dict()
        completed_objects = {
            "object": None,
            "type": 'pending',
            "mark_ends": 0,
            "frame_list": [],
            "frame_offsets": [],
            "frame_pulse": -1
        }

        self.frame_id += 1
        self.frame_list.append(self.frame_id)
        frame_list_output = []
        frame_offsets_output = []
        if self.buffer is not None:
            print(self.buffer.shape)
            self.buffer = np.vstack([self.buffer, frame])
        else:
            self.buffer = frame
        # self.buffer.append(frame)
        total_rows = self._buffer_total_rows()
        self.frame_offsets.append([
            self.current_frame_in_obj_start,
            self.current_frame_in_obj_start + frame.shape[0]
        ])
        self.current_frame_in_obj_start += frame.shape[0]
        print(f"total_rows:{total_rows}")
        logger.debug(f"[READ FRAME] 当前缓冲区总行数: {total_rows}, 模式: {self.mode}")
        # pdb.set_trace()
        if self.mode == "seek_mark":
            mark_exists, mark_nums, mark_ends = self.seek_mark(
                self._buffer_to_array(), self.frame_width_P,
                self.mark_length_P)
            logger.debug(f"[SEEK MARK] result: {mark_exists}")
            frame_pulse_obtained = False
            if mark_exists:
                logger.info(f"料尾找到！总共有{mark_ends[0]}行！")
                if self.input_type == "not_aligned":
                    if mark_ends[0] - self.mark_length_P > 2300:

                        first_object = self._extract_rows(
                            0,
                            mark_ends[0] - self.mark_length_P + self.overlap_P)
                        self._remove_rows(mark_ends[0] - self.mark_length_P -
                                          self.overlap_P)
                        completed_objects["object"] = first_object
                        completed_objects["type"] = "first_object"
                        self.mode = "check_mark"

                        frame_list_output = self.frame_list
                        frame_offsets_output = self.frame_offsets
                        completed_objects["frame_list"] = frame_list_output
                        completed_objects["frame_offsets"] = frame_offsets_output
                        self.frame_list = self.frame_list[-1:]
                        self.frame_offsets = []
                        lines_remain = self._buffer_total_rows()
                        self.current_frame_in_obj_start = lines_remain - frame.shape[
                            0]
                        self.frame_offsets.append([
                            self.current_frame_in_obj_start,
                            self.current_frame_in_obj_start + frame.shape[0]
                        ])
                        self.current_frame_in_obj_start += frame.shape[0]

                        return completed_objects
                else:
                    if mark_ends[0] > 2300:
                        first_object = self._extract_rows(
                            0, mark_ends[0] + self.overlap_P)
                        self._remove_rows(mark_ends[0] - self.overlap_P)
                        completed_objects["object"] = first_object
                        completed_objects["type"] = "first_object"
                        self.mode = "check_mark"

                        frame_list_output = self.frame_list
                        frame_offsets_output = self.frame_offsets
                        completed_objects["frame_list"] = frame_list_output
                        completed_objects["frame_offsets"] = frame_offsets_output
                        self.frame_list = self.frame_list[-1:]
                        self.frame_offsets = []
                        lines_remain = self._buffer_total_rows()
                        self.current_frame_in_obj_start = lines_remain - frame.shape[
                            0]
                        self.frame_offsets.append([
                            self.current_frame_in_obj_start,
                            self.current_frame_in_obj_start + frame.shape[0]
                        ])
                        self.current_frame_in_obj_start += frame.shape[0]

                        return completed_objects

                if not frame_pulse_obtained:
                    frame_pulse_output = frame_pulse - self.mark_to_end * CAM_PULSE_PER_PIXEL
                    frame_pulse_obtained = True
                if total_rows >= mark_ends[
                        0] + self.mark_to_end + self.overlap_P:
                    # mark_ends[0] + self.mark_to_end 是该mark对应的一个完整物料的长度
                    obj = self._extract_rows(
                        0, mark_ends[0] + self.mark_to_end + self.overlap_P)
                    self._remove_rows(mark_ends[0] + self.mark_to_end -
                                      self.overlap_P)

                    if mark_ends[
                            0] + self.mark_to_end - self.overlap_P < total_rows:
                        # 需要把组成这幅图的帧序号输出
                        frame_list_output = self.frame_list
                        # 需要把每个帧在大图的相对位置输出
                        frame_offsets_output = self.frame_offsets
                        self.frame_list = self.frame_list[-1:]
                        # 最后一帧还在buffer里， 比如 1,2 和3的一部分组成了一张图，3仍然在frame_list里
                        self.frame_offsets = []
                        self.current_frame_in_obj_start = \
                            total_rows - \
                            (mark_ends[0] - self.overlap_P + self.mark_to_end) - \
                            frame.shape[0]
                        self.frame_offsets.append([
                            self.current_frame_in_obj_start,
                            self.current_frame_in_obj_start + frame.shape[0]
                        ])
                        self.current_frame_in_obj_start += frame.shape[0]
                    else:
                        self.frame_list = []
                        self.current_frame_in_obj_start = 0
                        frame_offsets_output = self.frame_offsets
                        self.frame_offsets = []

                    completed_objects = {
                        "object": obj,
                        "type": "success",
                        "mark_ends": mark_ends,
                        "frame_list": frame_list_output,
                        "frame_offsets": frame_offsets_output,
                        "frame_pulse": frame_pulse_output
                    }
                    self.mode = "check_mark"
                else:
                    # TODO 还需要考虑下一帧也有mark， 检测到下一个mark的情况
                    logger.debug(f"[SEEK MARK] 缓冲区不足，等待更多帧。")
        elif self.mode == "check_mark":
            logger.error(
                f"block_size:{self.block_size} total_rows:{total_rows}")
            if total_rows >= self.block_size:
                block = self._extract_rows(0, self.block_size)
                margin = self.overlap_P // 2  # 用于扩大ROI，判断是否有空箔
                mark_exists, mark_end = self.check_mark(
                    block, self.frame_width_P, self.mark_length_P,
                    self.mark_roi, margin)
                lines_remain = total_rows - self.block_size
                block_pulse = frame_pulse - lines_remain * CAM_PULSE_PER_PIXEL
                frame_pulse_output = block_pulse - (
                    self.block_size - mark_end) * CAM_PULSE_PER_PIXEL

                if mark_exists:
                    standard_mark_end = self.block_size - self.overlap_P - self.mark_to_end
                    shift = mark_end - standard_mark_end
                    logger.debug(
                        f"[CHECK MARK]: mark end: {mark_end}, standard mark end: {standard_mark_end}"
                    )
                    logger.debug(f"[CHECK MARK] 偏移量: {shift}")
                    block = self._extract_rows(shift, self.block_size + shift)
                    self._remove_rows(self.block_size + shift -
                                      self.overlap_P * 2)
                    if total_rows > self.block_size + shift - self.overlap_P * 2:
                        frame_offsets_output = self.frame_offsets
                        frame_list_output = self.frame_list
                        self.frame_list = self.frame_list[-1:]  # 最后一张还在buffer里
                        self.frame_offsets = []
                        self.current_frame_in_obj_start = \
                            total_rows - \
                            (self.block_size + shift - self.overlap_P * 2) - \
                            frame.shape[0]
                        self.frame_offsets.append([
                            self.current_frame_in_obj_start,
                            self.current_frame_in_obj_start + frame.shape[0]
                        ])
                        self.current_frame_in_obj_start += frame.shape[0]
                    else:
                        frame_offsets_output = self.frame_offsets
                        frame_list_output = self.frame_list
                        self.frame_list = []
                        self.frame_offsets = []
                        self.current_frame_in_obj_start = 0
                    completed_objects = {
                        "object": block,
                        "type": "success",
                        "mark_ends": mark_end,
                        "frame_list": frame_list_output,
                        "frame_offsets": frame_offsets_output,
                        "frame_pulse": frame_pulse_output
                    }
                    logger.debug(
                        f"[CHECK MARK]: 输出 正常图，固定块: 0 ~ {self.block_size}")
                else:
                    self.problematic_pending = True
                    completed_objects = {
                        "object": block,
                        "type": "fail",
                        "mark_ends": -np.inf,
                        "frame_list": frame_list_output,
                        "frame_offsets": frame_offsets_output
                    }
                    logger.debug(f"[CHECK MARK]: 固定块检测失败，定义为问题图:")
                    self.mode = "seek_mark"
            else:
                logger.debug(f"[CHECK MARK] ：缓冲区不足，等待更多帧。")
        return completed_objects

    # def _buffer_to_array(self):
    #     """
    #     Concatenate all frames in the buffer to a single array.

    #     Returns:
    #         np.ndarray: A concatenated array of all frames in the buffer.
    #     """

    #     return np.concatenate(self.buffer, axis=0)

    def _buffer_to_array(self):
        return self.buffer

    # def _buffer_total_rows(self):
    #     """
    #     Calculate the total number of rows in all frames stored in the buffer.

    #     This method iterates over each frame in the buffer and sums up the number
    #     of rows (shape[0]) for each frame.

    #     Returns:
    #         int: The total number of rows in all frames in the buffer.
    #     """
    #     return sum(frame.shape[0] for frame in self.buffer)

    def _buffer_total_rows(self):
        return self.buffer.shape[0]

    # def _extract_rows(self, start, end):
    #     """
    #     Extracts and concatenates rows from frames in the buffer within the specified range.
    #
    #     Args:
    #         start (int): The starting row index (inclusive).
    #         end (int): The ending row index (exclusive).
    #
    #     Returns:
    #         np.ndarray or None: A concatenated array of the extracted rows if any rows are within the range,
    #                             otherwise None.
    #     """
    #     total = 0
    #     segments = []
    #     for frame in self.buffer:
    #         rows = frame.shape[0]
    #         if total + rows <= start:
    #             total += rows
    #             continue
    #         s = max(0, start - total)
    #         e = min(rows, end - total)
    #         segments.append(frame[s:e])
    #         total += rows
    #         if total >= end:
    #             break
    #     return np.concatenate(segments, axis=0) if segments else None

    def _extract_rows(self, start, end):
        if start < 0:
            start = 0
        if end > self._buffer_total_rows():
            end = self._buffer_total_rows()
        block = self.buffer[start:end, :]
        return block

    # def _remove_rows(self, num_rows):
    #     """
    #     Removes a specified number of rows from the beginning of each frame in the buffer.

    #     Args:
    #         num_rows (int): The number of rows to remove from the buffer.

    #     The method iterates through each frame in the buffer and removes the specified number of rows.
    #     If the number of rows to remove is greater than or equal to the number of rows in a frame,
    #     the entire frame is skipped. The remaining frames are adjusted accordingly and stored back
    #     in the buffer.
    #     """
    #     rows_to_remove = num_rows
    #     new_buffer = []
    #     for frame in self.buffer:
    #         rows = frame.shape[0]
    #         if rows_to_remove >= rows:
    #             rows_to_remove -= rows
    #             continue
    #         new_buffer.append(frame[rows_to_remove:])
    #         rows_to_remove = 0
    #     self.buffer = new_buffer

    def _remove_rows(self, num_rows):
        _buffer = self.buffer[num_rows:, :]
        self.buffer = _buffer

    def _squeeze_if_three_channels(self, image):
        return image[:, :, 0] if len(image.shape) == 3 else image

    def process_frame_timesai_tool(self, *args):
        """
        args[0]: jp_length_MM
        args[1]: mark_length_MM
        args[2]: input_type
        args[3]: frame
        args[4]: frame_pulse
        args[5]: cam_MM_per_P_Y
        args[6]: reset_signal

        """
        """
        ret_dict['5'] 是帧组->拼成大图的帧序号列表
        如果为[87, 65, 73, 84] (ascii的'WAIT')，表示为小图输出模式，
        小图输出模式输出给模型检测，此时下一个极片结果合并控件不合并结果，只储存模型给的矩形框

        ret_dict['6']是帧偏移 -> 每一帧在大图的位置[y_start, y_end]
         第一帧的y_start只可能为0或负数, 假设为-a, 那么这张图前a行不在此图里, 在上一张图中
        """
        ret_dict = {
            '0': 0,  # 工具状态
            '1': 'error',  # msg
            '2': 0,  # 空箔涂膏交界处坐标
            '3': [],  # 空箔涂膏交界处直线
            '4': [],  # 大图
            '5': [],
            '6': [],
            '7': [],  # 小图
            '8': [],  # 小图ID
            '9': [],  # 重叠区大小overlap
            '10': [],  # 涂膏空箔交叠处脉冲
            '11': 999,
        }
        try:
            args = args[0]
            # jp_length_MM = args[0]
            # mark_length_MM = args[1]
            jp_upside_type = args[0]
            jp_dimensions = args[1]
            input_type_mapping = {"对齐端": "aligned", "非对齐端": "not_aligned"}
            input_type = input_type_mapping[args[2]]
            frame = copy.deepcopy(args[3][0])
            frame = self._squeeze_if_three_channels(frame)

            frame_pulse = args[4][0]

            logger.error(args)

            cam_MM_per_P_Y = args[5][0][0]
            if args[6] is not None:
                reset_signal = args[6]
            else:
                reset_signal = 0

            # cam_loc = args[7]  # "up" or "down"
            """
                    up_CAM      down_CAM
            C_up     4,6          0,2
            C_down   0,2          4,6
            
            TODO: 后面考虑异或来做，可读性差
            """

            # jp_dimensions_map = {
            #     "up": {
            #         "C面在上": {
            #             "jp_length_MM": jp_dimensions[4],
            #             "mark_length_MM": jp_dimensions[6]
            #         },
            #         "C面在下": {
            #             "jp_length_MM": jp_dimensions[0],
            #             "mark_length_MM": jp_dimensions[2]
            #         },
            #     },
            #     "down": {
            #         "C面在上": {
            #             "jp_length_MM": jp_dimensions[0],
            #             "mark_length_MM": jp_dimensions[2]
            #         },
            #         "C面在下": {
            #             "jp_length_MM": jp_dimensions[4],
            #             "mark_length_MM": jp_dimensions[6]
            #         },
            #     }
            # }
            # logger.error(f"cam_loc:{cam_loc}, jp_upside_type:{jp_upside_type}")

            # jp_length_MM = jp_dimensions_map[cam_loc][jp_upside_type][
            #     "jp_length_MM"]
            # mark_length_MM = jp_dimensions_map[cam_loc][jp_upside_type][
            #     "mark_length_MM"]

            jp_length_MM = jp_dimensions[0]
            mark_length_MM = jp_dimensions[2]

            logger.error(
                f"jp_length_MM:{jp_length_MM}, mark_length_MM:{mark_length_MM}"
            )

            frame_width_P = frame.shape[1]

            overlap_MM = 50  # 100 mm
            if overlap_MM > mark_length_MM // 3:
                overlap_MM = mark_length_MM // 3
            overlap_P = int(round(overlap_MM / cam_MM_per_P_Y))
            mark_seeker = KongboSeeker()
            if not self.initialised:
                self._init_timesai_tool(mark_seeker, jp_length_MM,
                                        mark_length_MM, cam_MM_per_P_Y,
                                        frame_width_P, input_type, overlap_P)
            if reset_signal == 5:
                self._init_timesai_tool(mark_seeker, jp_length_MM,
                                        mark_length_MM, cam_MM_per_P_Y,
                                        frame_width_P, input_type, overlap_P)
                ret_dict["11"] = 0

            completed_objects = self.process_frame(frame, frame_pulse)
            ret_dict["1"] = completed_objects["type"]
            if ("success" in completed_objects["type"]) or (
                    "first_object" in completed_objects["type"]):
                ret_dict["0"] = 0

                if isinstance(completed_objects["mark_ends"], list):
                    ret_dict["2"] = completed_objects["mark_ends"][0]
                    mark_loc = float(completed_objects['mark_ends'][0])
                else:
                    ret_dict["2"] = completed_objects["mark_ends"]
                    mark_loc = float(completed_objects['mark_ends'])

                frame_pulse = completed_objects["frame_pulse"]
                if input_type == "aligned":
                    mark_loc = mark_loc - self.mark_length_P
                    frame_pulse = frame_pulse - self.mark_length_P * CAM_PULSE_PER_PIXEL
                    ret_dict["2"] = mark_loc
                ret_dict["3"] = [[[
                    1, 0.0, mark_loc,
                    float(self.frame_width_P), mark_loc
                ]]]
                ret_dict["4"].append({'image': completed_objects["object"]})
                ret_dict["5"] = completed_objects["frame_list"]
                ret_dict["6"] = completed_objects["frame_offsets"]
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                ret_dict["7"].append({'image': frame})
                ret_dict["8"] = self.frame_id
                ret_dict["9"] = self.overlap_P
                ret_dict["10"] = frame_pulse
            else:
                ret_dict["0"] = 1
                ret_dict["2"] = 0
                ret_dict["3"] = [[[1.0, 0.0, 0.0, 0.0, 0.0]]]
                ret_dict["4"].append({'image': None})
                ret_dict["5"] = [87, 65, 73, 84]  # W A I T
                ret_dict["6"] = [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                ret_dict["7"].append({'image': frame})
                ret_dict["8"] = self.frame_id
                ret_dict["9"] = self.overlap_P
                ret_dict["10"] = -1

        except Exception as e:
            traceback.print_exc()
            ret_dict["1"] = str(e)
        return ret_dict
