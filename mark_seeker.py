import pdb
import traceback
import cv2
from numpy import inf
from logging_config import logger
from abc import ABC, abstractmethod


class MarkSeeker(ABC):

    @abstractmethod
    def seek_mark():
        pass

    @abstractmethod
    def check_mark():
        pass


class KongboSeeker(MarkSeeker):

    def __init__(self):
        super().__init__()

    def seek_mark(self, buffer, obj_width, mark_length, ratio=0.05, mode="seek_mark", visiable=False):


        if ratio != 1.0:
            scaled_w = int(buffer.shape[1] * ratio)
            scaled_h = int(buffer.shape[0] * ratio)
            _image = cv2.resize(buffer, (scaled_w, scaled_h))
        else:
            _image = buffer

        if visiable:
            cv2.imshow('image', _image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # 根据缩放比例调整相关参数
        if ratio != 1.0:
            obj_width = int(obj_width * ratio)
            mark_length = int(mark_length * ratio)

        cut_edge_coefficient = 0.3  # 剪切系数
        margin = round(cut_edge_coefficient * obj_width)
        if margin == 0:
            margin = 1

        # 只看中间部分，避免边缘的干扰，也可以在前处理进行处理，那这里就不需要了
        roi = _image[:, margin:-margin]

        if visiable:
            cv2.imshow('roi', roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if len(roi.shape) == 3:
            if roi.shape[2] == 3:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)[1]
        if visiable:
            cv2.imshow('binary', binary)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8, ltype=cv2.CV_32S)
        max_area = 0
        max_label = 0
        valid_labels = []
        threshold = mark_length * obj_width * \
                    (1 - 2 * cut_edge_coefficient) * 0.8

        # 从 1 开始遍历（假设 0 为背景）
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            logger.debug(
                f"threshold: {threshold}, area: {area}, label: {label}")
            if area > threshold:
                valid_labels.append(label)
                if area > max_area:
                    max_area = area
                    max_label = label

        if len(valid_labels) == 0:
            return False, 0, [inf], [-inf]

        if mode == "seek_mark":  # 如果是seek mark模式，需要把所有的mark点位置返回
            pass
        elif mode == "check_mark":  # 如果是check mark模式，只返回最大的mark点位置，其实这里逻辑不算完备，但是先这么做
            valid_labels = [max_label]

        valid_mark_nums = len(valid_labels)
        logger.debug(f"valid_mark_nums: {valid_mark_nums}")
        valid_mark_ends = []
        valid_mark_starts = []
        for label in valid_labels:
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]

            left = x
            top = y
            right = x + w - 1
            bottom = y + h - 1
            if ratio != 1.0:
                bottom = int(bottom / ratio)
                top = int(top / ratio)
            valid_mark_ends.append(bottom)
            valid_mark_starts.append(top)
        logger.debug(
            f"valid_mark_nums: {valid_mark_nums} valid_mark_ends:{valid_mark_ends}")

        return True, valid_mark_nums, valid_mark_starts, valid_mark_ends

    def check_mark(self, block, obj_width, mark_length, mark_roi, margin, ratio=0.05, visiable=False):
        """
        Checks for the presence of a mark within a specified region of interest (ROI) in a given block.
        Args:
            block (numpy.ndarray): The image block in which to search for the mark.
            obj_width (int): The width of the object to be detected.
            mark_length (int): The length of the mark to be detected.
            mark_roi (tuple): A tuple (x, y, w, h) defining the region of interest where the mark is expected.
            margin (int): The margin to be added to the top and bottom of the ROI.
            ratio (float, optional): The ratio parameter for mark detection. Default is 0.05.
            visiable (bool, optional): If True, visualization of the mark detection process will be shown. Default is False.
        Returns:
            tuple: A tuple containing:
                - mark_exists (bool): True if the mark is found, False otherwise.
                - mark_end_in_block (int): The y-coordinate of the end of the mark within the block.
        """

        x, y, w, h = mark_roi
        start_x = x
        end_x = x + w
        start_y = y - margin
        end_y = y + h + margin
        roi = block[start_y:end_y, start_x:end_x]
        mark_exists, _, mark_start, mark_end = self.seek_mark(
            roi, obj_width, mark_length, ratio, mode="check_mark", visiable=visiable)
        mark_start_in_block = start_y + mark_start[-1]
        mark_end_in_block = start_y + mark_end[-1]
        return mark_exists, mark_start_in_block, mark_end_in_block
