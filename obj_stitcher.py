import pdb
from logging_config import SingletonLogger
import copy
import traceback
import cv2
import numpy as np
from numpy import inf
from mark_seeker import KongboSeeker

__DEBUG__ = False
CAM_MUL_FACTOR = 1
CAM_DIV_FACTOR = 1
FRAME_LINES = 5000
CAM_PULSE_PER_PIXEL = CAM_DIV_FACTOR / CAM_MUL_FACTOR
DISTANCE_BETWEEN_CAMS = 1950

__CALM__ = False


class ObjStitcher:

    def __init__(self):
        self.initialised = False

    def _ROAR(self, msg):
        if not __CALM__:
            self.logger.debug(msg)

    def get_simple_config_info(self) -> str:
        """获取简化的配置信息（纯文本格式）"""
        lines = [
            "===== 配置参数 =====",
            f"• 主相机: {'是' if self.is_major_camera else '否'}",
            f"• 像素精度: {self.cam_MM_per_P_Y} mm/像素",
            f"• 物料总长: {self.object_length_MM}mm (极片 {self.object_length_MM-self.mark_length_MM}mm + 空箔 {self.mark_length_MM}mm)",
            f"• 像素长度: 总长 {self.object_length_P}px | 极片 {self.jp_length_P}px | 空箔 {self.mark_length_P}px",
            f"• 入料方式: {self.input_type}",
            f"• 空箔位置: 末端 {self.mark_to_end}px",
            f"• 重叠区域: {self.overlap_P}px",
            f"• ROI边距: {self.roi_margin}px",
            f"• 输出图块: {self.block_size}px",
            f"• 帧宽度: {self.frame_width_P}px",
            f"• 空箔ROI: {self.mark_roi}"
        ]
        return "\n".join(lines)

    def get_simple_status_info(self) -> str:
        """获取简化的状态信息（纯文本格式）"""
        buffer_status = "无" if self.buffer is None else f"{len(self.buffer)}帧"
        frames_collected = f"{len(self.frame_list)}帧 (当前ID: {self.frame_id})"

        lines = [
            "\n===== 运行状态 =====",
            f"• 缓冲区: {buffer_status}",
            f"• 模式: {self.mode}",
            f"• 已收集帧: {frames_collected}"
        ]
        return "\n".join(lines)

    def get_simple_debug_info(self) -> str:
        """获取完整的简化调试信息"""
        config = self.get_simple_config_info()
        status = self.get_simple_status_info()

        logic_notes = [
            "\n===== 逻辑说明 =====",
            f"1. 入料方式: {'对齐(aligned)' if self.input_type == 'aligned' else '非对齐(not_aligned)'}",
            f"2. ROI基准: {'极片末端' if self.input_type == 'aligned' else '图像末端'}",
            f"3. 缓冲区: {'实时拼接' if self.buffer is None else '列表缓存'}"
        ]

        return "\n".join([config, status, *logic_notes])

    def _init_timesai_tool(self, is_major_camera, jp_length_MM, mark_length_MM,
                           cam_MM_per_P_Y, frame_width_P, input_type,
                           overlap_P, roi_margin, roi_binary_thresh):

        self.is_major_camera = is_major_camera
        singleton = SingletonLogger()
        if self.is_major_camera:
            logger_instance_name = "ObjStitcherMajorCamera"
        else:
            logger_instance_name = "ObjStitcherSecondaryCamera"
        self.logger = singleton.get_instance_logger(logger_instance_name)
        # 定义了一个找mark类，如果是其他种类的膜类，找料边界的核心方法可以在外面改
        self.mark_seeker = KongboSeeker(self.logger)
        self.seek_mark: function = self.mark_seeker.seek_mark
        self.check_mark: function = self.mark_seeker.check_mark

        # 在没有开始固定行取图，判断是否存在mark模式之前，是找mark模式
        self.mode = "SEEK MARK"  # 初始为seek mark模式

        # 整个物料的物理长度，单位mm
        self.mark_length_MM = mark_length_MM
        self.jp_length_MM = jp_length_MM
        self.object_length_MM = jp_length_MM + mark_length_MM

        # 像素精度
        self.cam_MM_per_P_Y = cam_MM_per_P_Y

        # 换算出来的整个物料的像素长度
        self.object_length_P = int(
            round(self.object_length_MM / cam_MM_per_P_Y))
        # 空箔像素长度
        self.mark_length_P = int(round(mark_length_MM / cam_MM_per_P_Y))
        # 极片像素长度
        self.jp_length_P = self.object_length_P - self.mark_length_P

        # 帧宽度 ~ 8K
        self.frame_width_P = frame_width_P

        # 入料方式， aligned 或者是 not_aligned
        # TODO 这个要和其他人写的代码统一叫法
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
            self.process_frame = self._process_frame_aligned
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
            self.process_frame = self._process_frame_not_aligned
        else:
            raise ValueError("input type must be 'aligned' or 'not_aligned'")

        # 每次输出物料时，物料上方和下方留的像素数量，便于找物料边缘，
        # 同时能解决一部分因为物料长度公差引起的找不到空箔的情况
        self.overlap_P = overlap_P
        self.roi_margin = roi_margin
        # 输出的图大小，等于完整物料加上 上方和下方两块overlap的大小
        self.block_size = self.object_length_P + 2 * self.overlap_P
        # 找空箔的roi，很显然这个和入料方式有关，与入料方式的关系体现在上面
        # self.mark_to_end的赋值上了
        self.mark_roi = [
            0, self.jp_length_P + self.overlap_P - self.mark_to_end,
            self.frame_width_P, self.mark_length_P
        ]

        if self.input_type == "not_aligned":
            self.mark_roi[1] += self.object_length_P

        self.buffer = None  # 存放连续帧

        # 为了后面的结果合并，需要把每一片料（每一个EA）由哪些帧组成的（frame_list)
        # 每一帧在这个完整物料中的相对位置（frame_offsets）保存起来
        self.frame_id = 0
        self.frame_list = []  # 每一片料由哪些帧组成
        self.frame_offsets = []
        self.frame_head_index_in_object = 0
        self.frame_lines = FRAME_LINES
        self.need_to_find_first_object = True
        self.problematic_pending = False
        # 初始化flag，True表示初始化已经完成
        self.initialised = True  # 初始化成功

        self.logger.debug(self.get_simple_config_info())

    def _buffer_to_array(self):
        return self.buffer

    def _buffer_total_rows(self):
        return self.buffer.shape[0]

    def _extract_rows(self, start, end):
        if start < 0:
            start = 0
        if end > self._buffer_total_rows():
            end = self._buffer_total_rows()
        block = self.buffer[start:end, :]
        return block

    def _remove_rows(self, num_rows):
        _buffer = self.buffer[num_rows:, :]
        self.buffer = _buffer

    def _squeeze_if_three_channels(self, image):
        return image[:, :, 0] if len(image.shape) == 3 else image

    def _process_buffer_to_output(self, output_size, shift, mark_start, mark_end, frame_pulse):
        # frame_pulse一定要是当前buffer尾的frame pulse
        # mark_start, mark_end 也一定要是相对于buffer的坐标
        # 初始化返回字典
        completed_object = {
            "object": None,
            "type": 'semi-finished',
            "mark_start": mark_start,
            "mark_end": mark_end,
            "mark_start_line": [],
            "mark_end_line": [],
            "mark_start_pulse": -1,
            "mark_end_pulse": -1,
            "frame_list": [],
            "frame_offsets": [],
            "roi": None,
        }
        self.logger.debug(
            f"[{self.mode}]preparing to output object --- {output_size} rows with {shift} shifts")
        obj = self._extract_rows(
            0 + shift, output_size + shift)
        actual_output_size = obj.shape[1]
        self._remove_rows(output_size - 2*self.overlap_P + shift)

        # 需要把组成这幅图的帧序号输出，这里多一点没关系，因为可以用frame_offsets来判断
        # 如果frame_list的下标对应的frame_offsets帧偏移大于这张图的大小，这一帧不会在结果合并里被考虑
        _frame_list = self.frame_list
        # 需要把每个帧在大图的相对位置输出
        _frame_offsets = self.frame_offsets
        if shift > 0:
            for i in range(len(_frame_offsets)):
                _frame_offsets[i][0] -= shift
                _frame_offsets[i][1] -= shift
            mark_end -= shift
            mark_start -= shift

        completed_object['mark_end'] = mark_end
        completed_object['mark_start'] = mark_start
        completed_object['mark_start_line'] = [
            1.0, 0.0, mark_start, self.buffer.shape[1], mark_start, 0.0]
        completed_object['mark_end_line'] = [
            1.0, 0.0, mark_end, self.buffer.shape[1], mark_end, 0.0]
        lines_remaining = self._buffer_total_rows()
        completed_object['mark_start_pulse'] = frame_pulse - \
            (output_size - mark_start + lines_remaining)*CAM_PULSE_PER_PIXEL
        completed_object['mark_end_pulse'] = frame_pulse - \
            (output_size - mark_end + lines_remaining)*CAM_PULSE_PER_PIXEL

        # if there are still lines in the buffer, we need to update the frame_list and frame_offsets
        # aparently there are, otherwise this function would not be called
        # I, vchacha, wrote the following condition sentence to improve the readability of the code
        if self._buffer_total_rows() >= 0:
            nof_frames_to_keep = self._buffer_total_rows(
            )//self.frame_lines + 1
            self.frame_list = self.frame_list[-nof_frames_to_keep:]

            self.frame_head_index_in_object = self._buffer_total_rows() - self.frame_lines * \
                nof_frames_to_keep

            self.frame_offsets = []

            for _ in range(nof_frames_to_keep):
                self.frame_offsets.append([
                    self.frame_head_index_in_object,
                    self.frame_head_index_in_object +
                    self.frame_lines - 1
                ])
                self.frame_head_index_in_object += self.frame_lines
            completed_object['frame_list'] = _frame_list
            completed_object['frame_offsets'] = _frame_offsets
            completed_object['object'] = obj
        else:
            self.logger.warning(
                f"[{self.mode}] Whoops, you find the ROOM OF REQUIREMENT")
        return completed_object

    def _need_to_output_seperately(self, mark_end):
        if self.input_type == "aligned":
            if mark_end < DISTANCE_BETWEEN_CAMS:
                return False
            else:
                return True
        elif self.input_type == "not_aligned":
            if mark_end - self.mark_length_P < DISTANCE_BETWEEN_CAMS:
                return False
            else:
                return True

    def _process_frame_aligned(self, frame, frame_pulse):
        if self.buffer is not None:
            self.buffer = np.vstack([self.buffer, frame])
        else:
            self.buffer = frame
        self.frame_id += 1
        self.frame_list.append(self.frame_id)
        self.frame_offsets.append(
            [self.frame_head_index_in_object, self.frame_head_index_in_object + self.frame_lines - 1])
        self.frame_head_index_in_object += self.frame_lines
        self.logger.debug(self.get_simple_status_info())
        pending_object = {
            "object": None,
            "type": 'pending',
            "mark_start": [-1],
            "mark_end": [-1],
            "mark_start_line": [],
            "mark_end_line": [],
            "mark_start_pulse": -1,
            "mark_end_pulse": -1,
            "frame_list": [87, 65, 73, 84],
            "frame_offsets": [[-1, -1], [-1, -1], [-1, -1], [-1, -1]],
            "mark_start_pulse": -1,
            "mark_end_pulse": -1,
            "roi": None,
        }

        # =====================================================================#
        # 下面是SEEK MARK模式
        # =====================================================================#

        if self.mode == "SEEK MARK":
            ret_dict = self.seek_mark(
                buffer=self.buffer, standard_mark_length=self.mark_length_P, mark_area_ratio_thresh=0.8, mark_roi_binary_thresh=190, mark_length_tolarence=21, ratio=1)
            mark_exists = ret_dict["bottom_mark"]['mark_exists']
            mark_start = ret_dict["bottom_mark"]['mark_start']
            mark_end = ret_dict["bottom_mark"]['mark_end']
            mark_exists_in_buffer_tail = ret_dict["bottom_mark"]['in_tail']
            valid_mark_nums = ret_dict['valid_mark_nums']
            self.logger.debug(
                f"[{self.mode}]mark exists: {mark_exists}, valid_mark_nums: {valid_mark_nums}")

            if not mark_exists:
                self.logger.debug(
                    f"[{self.mode}]mark not found, waiting for next frame")
                if mark_exists_in_buffer_tail:
                    self.logger.debug(
                        f"[{self.mode}]mark partially exists in buffer tail, waiting for next frame")
                return pending_object
            if mark_exists:
                self.logger.debug(
                    f"[{self.mode}]mark found, mark start: {mark_start}, mark end: {mark_end}, buffer size: {self._buffer_total_rows()}, overlap: {self.overlap_P}")
                output_size = mark_end + self.overlap_P
                if self._buffer_total_rows() < output_size:
                    self.logger.debug(
                        f"[{self.mode}]not enough rows in buffer, waiting for next frame --- {output_size - self._buffer_total_rows()} lines required")
                    return pending_object

                if self.is_major_camera and self.need_to_find_first_object:
                    self.logger.debug(f"[{self.mode}]processing first object")
                    if not self._need_to_output_seperately(mark_end):
                        self.logger.debug(
                            f"[{self.mode}]first object merged with the coming object")
                        self.need_to_find_first_object = False
                        return pending_object
                    else:
                        self.logger.debug(
                            f"[{self.mode}]output first object seperately")
                        self.need_to_find_first_object = False
                        object_type = "first_object"
                if not self.problematic_pending:
                    object_type = "first_object"
                else:
                    object_type = "problematic"
                completed_object = self._process_buffer_to_output(
                    output_size=output_size, shift=0, mark_start=mark_start, mark_end=mark_end, frame_pulse=frame_pulse)
                completed_object['type'] = object_type
                self.logger.debug(
                    f"[{self.mode}]status switched to check mark")
                self.mode = "CHECK MARK"
                return completed_object

        # =====================================================================#
        # 下面是CHECK MARK模式
        # =====================================================================#

        if self.mode == "CHECK MARK":
            if not self._buffer_total_rows() >= self.block_size:
                self.logger.debug(
                    f"[{self.mode}]waiting for next frame")
                return pending_object
            else:
                block = self._extract_rows(0, self.block_size)
                lines_remaining = self._buffer_total_rows()
                block_end_pulse = frame_pulse - lines_remaining*CAM_PULSE_PER_PIXEL  # not used
                margin = self.roi_margin  # 用于扩大ROI，判断是否有空箔
                ret_dict = self.check_mark(block=block, standard_mark_length=self.mark_length_P, mark_roi=self.mark_roi,
                                           margin=margin, mark_area_ratio_thresh=0.6, mark_roi_binary_thresh=190, mark_length_tolarence=10, ratio=1)
                mark_exists = ret_dict['mark_exists']
                mark_start = ret_dict['mark_start']
                mark_end = ret_dict['mark_end']
                mark_exists_in_buffer_tail = ret_dict['in_tail']
                self.logger.debug(
                    f"[{self.mode}]mark exists: {mark_exists}, in tail: {mark_exists_in_buffer_tail}")
                if not mark_exists:
                    self.logger.debug(
                        f"[{self.mode}]problematic object found, search if mark exists in buffer")
                    inner_ret_dict = self.seek_mark(
                        buffer=self.buffer, standard_mark_length=self.mark_length_P, mark_area_ratio_thresh=0.6, mark_roi_binary_thresh=190, mark_length_tolarence=21, ratio=1)
                    inner_mark_exists = inner_ret_dict["bottom_mark"]['mark_exists']
                    inner_mark_start = inner_ret_dict["bottom_mark"]['mark_start']
                    inner_mark_end = inner_ret_dict["bottom_mark"]['mark_end']
                    inner_mark_exists_in_buffer_tail = inner_ret_dict["bottom_mark"]['in_tail']
                    inner_valid_mark_nums = inner_ret_dict['valid_mark_nums']
                    self.logger.debug(
                        f"[{self.mode}]inner mark exists: {inner_mark_exists}, valid_mark_nums: {inner_valid_mark_nums}")
                    if not inner_mark_exists:
                        if inner_mark_exists_in_buffer_tail:
                            self.logger.debug(
                                f"[{self.mode}]mark partially exists in buffer tail, waiting for next frame")
                        self.logger.debug(
                            f"[{self.mode}]status switched to seek mark")
                        self.mode == "SEEK MARK"
                        self.problematic_pending = True
                        return pending_object
                    if inner_mark_exists:
                        output_size = inner_mark_end + self.overlap_P
                        self.logger.debug(
                            f"[{self.mode}]inner mark found, mark start: {inner_mark_start}, mark end: {inner_mark_end}, buffer size: {self._buffer_total_rows()}, overlap: {self.overlap_P}")
                        if self._buffer_total_rows() < output_size:
                            self.logger.debug(
                                f"[{self.mode}]not enough rows in buffer, waiting for next frame")
                            self.logger.debug(
                                f"[{self.mode}]status switched to seek mark")
                            self.mode == "SEEK MARK"
                            self.problematic_pending = True
                            return pending_object
                        else:
                            completed_object = self._process_buffer_to_output(
                                output_size=output_size, shift=0, mark_end=inner_mark_end, mark_start=inner_mark_start, frame_pulse=frame_pulse)
                            object_type = "problematic"
                            completed_object['type'] = object_type
                            self.logger.debug(
                                f"[{self.mode}]inner mark found and object outputed, status stays check mark")
                            self.mode = "CHECK MARK"
                            return completed_object

                if mark_exists:
                    standard_mark_end = self.block_size - self.overlap_P
                    shift = mark_end - standard_mark_end
                    self.logger.debug(
                        f"[{self.mode}]mark end: {mark_end}, standard mark end: {standard_mark_end}, shift: {shift}")
                    output_size = self.block_size + shift
                    if self._buffer_total_rows() < output_size:
                        self.logger.debug(
                            f"[{self.mode}]not enough rows in buffer, waiting for next frame")
                        return pending_object
                    else:
                        completed_object = self._process_buffer_to_output(
                            output_size=output_size, shift=shift, mark_end=mark_end, mark_start=mark_start, frame_pulse=frame_pulse)
                        object_type = "success"
                        completed_object['type'] = object_type
                        return completed_object

    def _process_frame_not_aligned(self, frame, frame_pulse):
        if self.buffer is not None:
            self.buffer = np.vstack([self.buffer, frame])
        else:
            self.buffer = frame
        self.frame_id += 1
        self.frame_list.append(self.frame_id)
        self.frame_offsets.append(
            [self.frame_head_index_in_object, self.frame_head_index_in_object + self.frame_lines - 1])
        self.frame_head_index_in_object += self.frame_lines
        self.logger.debug(self.get_simple_status_info())
        pending_object = {
            "object": None,
            "type": 'pending',
            "mark_start": [-1],
            "mark_end": [-1],
            "mark_start_line": [],
            "mark_end_line": [],
            "mark_start_pulse": -1,
            "mark_end_pulse": -1,
            "frame_list": [87, 65, 73, 84],
            "frame_offsets": [[-1, -1], [-1, -1], [-1, -1], [-1, -1]],
            "mark_start_pulse": -1,
            "mark_end_pulse": -1,
            "roi": None,
        }

        # =====================================================================#
        # 下面是SEEK MARK模式
        # =====================================================================#

        if self.mode == "SEEK MARK":
            ret_dict = self.seek_mark(
                buffer=self.buffer, standard_mark_length=self.mark_length_P, mark_area_ratio_thresh=0.8, mark_roi_binary_thresh=190, mark_length_tolarence=21, ratio=1)
            mark_exists = ret_dict["top_mark"]['mark_exists']
            mark_start = ret_dict["top_mark"]['mark_start']
            mark_end = ret_dict["top_mark"]['mark_end']
            mark_exists_in_buffer_tail = ret_dict["top_mark"]['in_tail']
            valid_mark_nums = ret_dict['valid_mark_nums']
            self.logger.debug(
                f"[{self.mode}]mark exists: {mark_exists}, valid_mark_nums: {valid_mark_nums}")

            if not mark_exists:
                self.logger.debug(
                    f"[{self.mode}]mark not found, waiting for next frame")
                if mark_exists_in_buffer_tail:
                    self.logger.debug(
                        f"[{self.mode}]mark partially exists in buffer tail, waiting for next frame")
                return pending_object
            if mark_exists:
                self.logger.debug(
                    f"[{self.mode}]mark found, mark start: {mark_start}, mark end: {mark_end}, buffer size: {self._buffer_total_rows()}, overlap: {self.overlap_P}")
                output_size = mark_start + self.overlap_P
                if self._buffer_total_rows() < output_size:
                    self.logger.debug(
                        f"[{self.mode}]not enough rows in buffer, waiting for next frame --- {output_size - self._buffer_total_rows()} lines required")
                    return pending_object

                if self.is_major_camera and self.need_to_find_first_object:
                    self.logger.debug(f"[{self.mode}]processing first object")
                    if not self._need_to_output_seperately(mark_end):
                        self.logger.debug(
                            f"[{self.mode}]first object merged with the coming object")
                        self.need_to_find_first_object = False
                        return pending_object
                    else:
                        self.logger.debug(
                            f"[{self.mode}]output first object seperately")
                        self.need_to_find_first_object = False
                        object_type = "first_object"
                if not self.problematic_pending:
                    object_type = "first_object"
                else:
                    object_type = "problematic"
                completed_object = self._process_buffer_to_output(
                    output_size=output_size, shift=0, mark_start=mark_start - self.object_length_P, mark_end=mark_end - self.object_length_P, frame_pulse=frame_pulse)
                completed_object['type'] = object_type
                self.logger.debug(
                    f"[{self.mode}]status switched to check mark")
                self.mode = "CHECK MARK"
                return completed_object

        # =====================================================================#
        # 下面是CHECK MARK模式
        # =====================================================================#

        if self.mode == "CHECK MARK":
            if not self._buffer_total_rows() >= self.block_size + self.mark_length_P:  # look one more mark
                self.logger.debug(
                    f"[{self.mode}], waiting for next frame")
                return pending_object
            else:
                block = self._extract_rows(
                    0, self.block_size + self.mark_length_P)
                lines_remaining = self._buffer_total_rows()
                block_end_pulse = frame_pulse - lines_remaining*CAM_PULSE_PER_PIXEL  # not used
                margin = self.roi_margin  # 用于扩大ROI，判断是否有空箔
                ret_dict = self.check_mark(block=block, standard_mark_length=self.mark_length_P, mark_roi=self.mark_roi,
                                           margin=margin, mark_area_ratio_thresh=0.6, mark_roi_binary_thresh=190, mark_length_tolarence=21, ratio=1)
                mark_exists = ret_dict['mark_exists']
                mark_start = ret_dict['mark_start']
                mark_end = ret_dict['mark_end']
                mark_exists_in_buffer_tail = ret_dict['in_tail']
                self.logger.debug(
                    f"[{self.mode}]mark exists: {mark_exists}, in tail: {mark_exists_in_buffer_tail}")
                if not mark_exists:
                    self.logger.debug(
                        f"[{self.mode}]problematic object found, search if mark exists in buffer")
                    inner_ret_dict = self.seek_mark(
                        buffer=self.buffer, standard_mark_length=self.mark_length_P, mark_area_ratio_thresh=0.6, mark_roi_binary_thresh=190, mark_length_tolarence=21, ratio=1)
                    inner_mark_exists = inner_ret_dict["bottom_mark"]['mark_exists']
                    inner_mark_start = inner_ret_dict["bottom_mark"]['mark_start']
                    inner_mark_end = inner_ret_dict["bottom_mark"]['mark_end']
                    inner_mark_exists_in_buffer_tail = inner_ret_dict["bottom_mark"]['in_tail']
                    inner_valid_mark_nums = inner_ret_dict['valid_mark_nums']
                    self.logger.debug(
                        f"[{self.mode}]inner mark exists: {inner_mark_exists}, valid_mark_nums: {inner_valid_mark_nums}")
                    if not inner_mark_exists:
                        if inner_mark_exists_in_buffer_tail:
                            self.logger.debug(
                                f"[{self.mode}]mark partially exists in buffer tail, waiting for next frame")
                        self.logger.debug(
                            f"[{self.mode}]status switched to seek mark")
                        self.mode == "SEEK MARK"
                        self.problematic_pending = True
                        return pending_object
                    if inner_mark_exists:
                        output_size = inner_mark_start + self.overlap_P
                        self.logger.debug(
                            f"[{self.mode}]inner mark found, mark start: {inner_mark_start}, mark end: {inner_mark_end}, buffer size: {self._buffer_total_rows()}, overlap: {self.overlap_P}")
                        if self._buffer_total_rows() < output_size:
                            self.logger.debug(
                                f"[{self.mode}]not enough rows in buffer, waiting for next frame")
                            self.logger.debug(
                                f"[{self.mode}]status switched to seek mark")
                            self.mode == "SEEK MARK"
                            self.problematic_pending = True
                            return pending_object
                        else:
                            completed_object = self._process_buffer_to_output(
                                output_size=output_size, shift=0, mark_end=inner_mark_end - self.object_length_P, mark_start=inner_mark_start - self.object_length_P, frame_pulse=frame_pulse)
                            object_type = "problematic"
                            completed_object['type'] = object_type
                            self.logger.debug(
                                f"[{self.mode}]inner mark found and object outputed, status stays check mark")
                            self.mode = "CHECK MARK"
                            return completed_object

                if mark_exists:
                    standard_mark_start = self.block_size - self.overlap_P
                    shift = mark_start - standard_mark_start
                    self.logger.debug(
                        f"[{self.mode}]mark end: {mark_start}, standard mark end: {standard_mark_start}, shift: {shift}")
                    output_size = self.block_size + shift
                    if self._buffer_total_rows() < output_size:
                        self.logger.debug(
                            f"[{self.mode}], not enough rows in buffer, waiting for next frame")
                        return pending_object
                    else:
                        completed_object = self._process_buffer_to_output(
                            output_size=output_size, shift=shift, mark_end=mark_end - self.object_length_P, mark_start=mark_start - self.object_length_P, frame_pulse=frame_pulse)
                        object_type = "success"
                        completed_object['type'] = object_type
                        return completed_object

    # the most ugly part
    def process_frame_timesai_tool(self, *args):
        """
        args[0]: is_major_camera  是/否
        args[1]: jp_dimensions
        args[2]: input_type   对齐端/非对齐端
        args[3]: frame
        args[4]: frame_pulse
        args[5]: cam_MM_per_P_Y  像素精度
        args[6]: reset_signal    复位信号
        args[7]: overlap_MM
        args[8]: roi_margin
        args[9]: roi_binary_thresh
        args[10]: output image options  缓存队列/输出图像/帧图/roi/不显示
        """
        ret_dict = {
            '0': 0,     # 工具状态码
            '1': 'error',  # 工具状态信息
            '2': [],    # 空箔涂膏交界处坐标
            '3': [],    # 空箔涂膏交界处直线
            '4': [],    # 大图
            '5': [],    # 组成大图的frame序号列表
            '6': [],    # 每一帧在大图的偏移
            '7': [],    # 小图
            '8': [],    # 小图ID
            '9': 0,     # 重叠区大小overlap
            '10': [],   # 涂膏空箔交叠处脉冲
            '11': 999,  # 复位状态码
            '12': [],   # 显示图像
        }

        args = args[0]
        try:
            is_major_camera = True if args[0] == "是" else False
            jp_dimensions = args[1]
            jp_length_MM = jp_dimensions[0]
            mark_length_MM = jp_dimensions[2]
            input_type_mapping = {"对齐端": "aligned", "非对齐端": "not_aligned"}
            input_type = input_type_mapping[args[2]]
            frame = copy.deepcopy(args[3][0])
            frame = self._squeeze_if_three_channels(frame)
            frame_width_P = frame.shape[1]

            frame_pulse = args[4][0]
            cam_MM_per_P_Y = args[5][0][0]
            if args[6] is not None:
                reset_signal = args[6]
            else:
                reset_signal = 0

            overlap_MM = args[7]
            # if overlap_MM > mark_length_MM // 3:
            #     overlap_MM = mark_length_MM // 3
            overlap_P = int(round(overlap_MM / cam_MM_per_P_Y))
            roi_margin = args[8]
            roi_binary_thresh = args[9]
            result_to_show = args[10]

            if not self.initialised:

                self._init_timesai_tool(is_major_camera, jp_length_MM,
                                        mark_length_MM, cam_MM_per_P_Y,
                                        frame_width_P, input_type, overlap_P, roi_margin, roi_binary_thresh)
            if reset_signal == 5:
                self._init_timesai_tool(is_major_camera, jp_length_MM,
                                        mark_length_MM, cam_MM_per_P_Y,
                                        frame_width_P, input_type, overlap_P, roi_margin, roi_binary_thresh)
                ret_dict['11'] = 0

            completed_object = self.process_frame(frame, frame_pulse)
            obj_type = completed_object['type']
            mark_start = completed_object['mark_start']
            mark_end = completed_object['mark_end']
            mark_start_line = completed_object['mark_start_line']
            mark_end_line = completed_object['mark_end_line']
            mark_start_pulse = completed_object['mark_start_pulse']
            mark_end_pulse = completed_object['mark_end_pulse']
            frame_list = completed_object['frame_list']
            frame_offsets = completed_object['frame_offsets']

            ret_dict['1'] = obj_type
            if ("success" in obj_type) or (
                    "first_object" in obj_type) or ("problematic" in obj_type):
                ret_dict['0'] = 0
                if input_type == "aligned":
                    ret_dict['2'].append([mark_start, mark_end])
                    ret_dict['3'].append([mark_start_line, mark_end_line])
                    ret_dict['10'].append([mark_start_pulse, mark_end_pulse])
                else:
                    ret_dict['2'].append([mark_end, mark_start])
                    ret_dict['3'].append([mark_end_line, mark_start_line])
                    ret_dict['10'].append([mark_end_pulse, mark_start_pulse])

                ret_img = completed_object['object']
                if isinstance(ret_img, tuple):  # 不知道为什么要加这个
                    ret_img = ret_img[0]
                ret_dict['4'].append({'image': ret_img})
                ret_dict['5'] = frame_list
                ret_dict['6'] = frame_offsets
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                ret_dict['7'].append({'image': frame})
                ret_dict['8'] = self.frame_id
                ret_dict['9'] = self.overlap_P
                # ret_dict['10'] = int(frame_pulse)
            else:
                ret_dict['0'] = 1
                ret_dict['2'] = [[0, 0]]
                ret_dict['3'] = [
                    [[1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]]]
                ret_dict['4'].append({'image': None})
                ret_dict['5'] = [87, 65, 73, 84]  # W A I T
                ret_dict['6'] = [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                ret_dict['7'].append({'image': frame})
                ret_dict['8'] = self.frame_id
                ret_dict['9'] = self.overlap_P
                ret_dict['10'] = [[-1, -1]]

        except Exception as e:
            traceback.print_exc()
            ret_dict['1'] = str(e)
        update_result_dict(ret_dict, result_to_show,
                           completed_object, self.buffer)

        self.logger.debug(format_ret_dict(ret_dict))

        return ret_dict


def update_result_dict(ret_dict: dict, result_to_show: str,
                       completed_object: dict = None,
                       buffer: np.ndarray = None) -> dict:
    """
    根据不同的显示类型更新结果字典

    Args:
        ret_dict: 要更新的结果字典
        result_to_show: 显示类型选项 ("roi"/"输出图像"/"帧图"/"缓存队列")
        completed_object: 包含ROI图像的对象 (仅result_to_show="roi"时需要)
        buffer_obj: 包含缓冲数据的对象 (仅result_to_show="缓存队列"时需要)

    Returns:
        更新后的结果字典
    """
    # 确保ret_dict['12']存在
    if '12' not in ret_dict:
        ret_dict['12'] = []

    if result_to_show == "roi":
        roi_image = completed_object.get("roi") if completed_object else None
        image_data = np.expand_dims(
            roi_image, axis=2) if roi_image is not None else None
        ret_dict['12'].append({'image': image_data})

    elif result_to_show == "输出图像":
        ret_dict['12'] = ret_dict.get('4', [])

    elif result_to_show == "帧图":
        ret_dict['12'] = ret_dict.get('7', [])

    elif result_to_show == "缓存队列" and buffer is not None:
        ret_dict['12'].append({'image': buffer})

    else:
        ret_dict['12'].append({'image': None})

    return ret_dict


def format_ret_dict(ret_dict):
    lines = [
        "\n===== 输出结果 =====",
        f"• {'状态码':<8}:\t {ret_dict['0']}",
        f"• {'状态信息':<8}:\t {ret_dict['1']}",
        f"• {'空箔涂膏交界处坐标':<12}:\t {ret_dict['2']}",
        f"• {'空箔涂膏交界处直线':<12}:\t {ret_dict['3']}",
        f"• {'组成大图的frame序号列表':<12}:\t {ret_dict['5']}",
        f"• {'每一帧在大图的偏移':<12}:\t {ret_dict['6']}",
        f"• {'小图ID':<8}:\t {ret_dict['8']}",
        f"• {'重叠区大小overlap':<12}:\t {ret_dict['9']}",
        f"• {'涂膏空箔交叠处脉冲':<12}:\t {ret_dict['10']}",
        f"• {'复位状态码':<8}:\t {ret_dict['11']}",
        f"• {'拼图大小':<8}:\t {get_image_shape_safe(ret_dict['4'])}",
    ]
    return "\n".join(lines)


def get_image_shape_safe(data):
    try:
        return data[0]['image'].shape if data and data[0].get('image') is not None else 'None'
    except (IndexError, TypeError, AttributeError, KeyError):
        return 'None'
