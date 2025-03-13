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

        cut_edge_coefficient = 0.2  # 剪切系数
        margin = round(cut_edge_coefficient * obj_width)
        if margin == 0:
            margin = 1

        # 只看中间部分，避免边缘的干扰，也可以在前处理进行处理，那这里就不需要了
        roi = _image[:, margin:-margin]

        if visiable:
            cv2.imshow('roi', roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

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
        threshold = mark_length * obj_width * (1-2*cut_edge_coefficient) * 0.95

        # 从 1 开始遍历（假设 0 为背景）
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area > threshold:
                valid_labels.append(label)
                logger.debug(
                    f"threshold: {threshold}, area: {area}, label: {label}")
                if area > max_area:
                    max_area = area
                    max_label = label

        if len(valid_labels) == 0:
            return False, 0, [-inf]

        if mode == "seek_mark":  # 如果是seek mark模式，需要把所有的mark点位置返回
            pass
        elif mode == "check_mark":  # 如果是check mark模式，只返回最大的mark点位置，其实这里逻辑不算完备，但是先这么做
            valid_labels = [max_label]

        valid_mark_nums = len(valid_labels)
        logger.debug(f"valid_mark_nums: {valid_mark_nums}")
        valid_mark_ends = []
        for label in valid_labels:
            x = stats[max_label, cv2.CC_STAT_LEFT]
            y = stats[max_label, cv2.CC_STAT_TOP]
            w = stats[max_label, cv2.CC_STAT_WIDTH]
            h = stats[max_label, cv2.CC_STAT_HEIGHT]

            left = x
            top = y
            right = x + w - 1
            bottom = y + h - 1
            if ratio != 1.0:
                bottom = int(bottom / ratio)
            valid_mark_ends.append(bottom)
        return True, valid_mark_nums, valid_mark_ends

    def check_mark(self, block, obj_width, mark_length, mark_roi, margin, ratio=0.05, visiable=False):
        x, y, w, h = mark_roi
        start_x = x
        end_x = x + w
        start_y = y - margin
        end_y = y + h + margin
        roi = block[start_y:end_y, start_x:end_x]
        mark_exists, _, mark_end = self.seek_mark(
            roi, obj_width, mark_length, ratio, mode="check_mark", visiable=visiable)
        mark_end_in_block = start_y + mark_end[-1]
        return mark_exists, mark_end_in_block
