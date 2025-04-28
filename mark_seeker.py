import pdb
import traceback
import cv2
from numpy import inf
import numpy
from logging_config import SingletonLogger
from abc import ABC, abstractmethod


class MarkSeeker(ABC):

    @abstractmethod
    def seek_mark(self):
        pass

    @abstractmethod
    def check_mark(self):
        pass


class KongboSeeker(MarkSeeker):

    def __init__(self, logger=None):
        if logger is None:
            singleton = SingletonLogger("KongboSeeker")
            self.logger = singleton.get_instance_logger("KongboSeeker")
        else:
            self.logger = logger

        super().__init__()

    def seek_mark(self, buffer, standard_mark_length, mark_area_ratio_thresh=0.8, mark_roi_binary_thresh=190, mark_length_tolarence=2, ratio=0.05):
        """
        Analyzes an image buffer to identify and classify marks based on their position 
        and area characteristics.
        Args:
            buffer (numpy.ndarray): The input image buffer, expected to be a 2D or 3D array.
            standard_mark_length (int): The standard length of a mark for comparison.
            mark_area_ratio_thresh (float, optional): The minimum area ratio threshold for 
                a valid mark. Defaults to 0.8.
            mark_roi_binary_thresh (int, optional): The binary threshold value for 
                converting the region of interest (ROI) to binary. Defaults to 190.
            mark_length_tolarence (int, optional): The tolerance for mark length when 
                determining if a mark is in the tail region. Defaults to 2.
            ratio (float, optional): The scaling ratio for resizing the image buffer. 
                Defaults to 0.05.
        Returns:
            dict: A dictionary containing the following keys:
                - "top_mark": Information about the topmost mark, including:
                    - "exists" (bool): Whether a top mark exists.
                    - "start" (int): The starting y-coordinate of the top mark.
                    - "end" (int): The ending y-coordinate of the top mark.
                    - "in_tail" (bool): Whether the top mark is in the tail region.
                - "bottom_mark": Information about the bottommost mark, including:
                    - "exists" (bool): Whether a bottom mark exists.
                    - "start" (int): The starting y-coordinate of the bottom mark.
                    - "end" (int): The ending y-coordinate of the bottom mark.
                    - "in_tail" (bool): Whether the bottom mark is in the tail region.
                - "best_area_mark": Information about the mark with the area closest to 
                  the threshold, including:
                    - "exists" (bool): Whether such a mark exists.
                    - "start" (int): The starting y-coordinate of the mark.
                    - "end" (int): The ending y-coordinate of the mark.
                    - "area_diff" (float): The absolute difference between the mark's 
                      area and the threshold area.
                    - "in_tail" (bool): Whether the mark is in the tail region.
                - "valid_mark_nums" (int): The total number of valid marks detected.
        """

        ret_dict = {
            "top_mark": {"mark_exists": False, "mark_start": -1, "mark_end": -1, "in_tail": False},
            "bottom_mark": {"mark_exists": False, "mark_start": -1, "mark_end": -1, "in_tail": False},
            "best_area_mark": {"mark_exists": False, "mark_start": -1, "mark_end": -1, "in_tail": False, "area_diff": float('inf')},
            "valid_mark_nums": 0,
        }

        buffer_width = buffer.shape[1]

        if ratio != 1.0:
            scaled_w = int(buffer.shape[1] * ratio)
            scaled_h = int(buffer.shape[0] * ratio)
            _image = cv2.resize(buffer, (scaled_w, scaled_h))
        else:
            _image = buffer

        # adjust buffer_width and mark_length based on the scaling ratio
        scaled_buffer_width = int(buffer_width * ratio)
        scaled_standard_mark_length = int(standard_mark_length * ratio)

        cut_edge_coefficient = 0.3  # 剪切系数
        margin = round(cut_edge_coefficient * scaled_buffer_width)
        margin = margin if margin > 0 else 1

        # concentrate on the center of the image
        roi = _image[:, margin:-margin]

        if len(roi.shape) == 3:
            if roi.shape[2] == 3:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(roi, mark_roi_binary_thresh,
                               255, cv2.THRESH_BINARY)[1]

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8, ltype=cv2.CV_32S)
        # Calculate area threshold
        mark_area_thresh = scaled_standard_mark_length * scaled_buffer_width * \
            (1 - 2 * cut_edge_coefficient) * mark_area_ratio_thresh

        # Track all valid marks
        valid_marks = []

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area > mark_area_thresh:
                x = stats[label, cv2.CC_STAT_LEFT]
                y = stats[label, cv2.CC_STAT_TOP]
                w = stats[label, cv2.CC_STAT_WIDTH]
                h = stats[label, cv2.CC_STAT_HEIGHT]

                # Convert back to original coordinates
                start = int(y / ratio)
                end = int((y + h) / ratio)

                valid_marks.append({
                    "label": label,
                    "mark_start": start,
                    "mark_end": end,
                    "mark_area": area,
                    "area_diff": abs(area - mark_area_thresh),
                })

        ret_dict["valid_mark_nums"] = len(valid_marks)

        if not valid_marks:
            return ret_dict

        # Find top most mark (minimum y position)
        top_mark = min(valid_marks, key=lambda m: m["mark_start"])
        ret_dict["top_mark"] = {
            "mark_exists": True,
            "mark_start": top_mark["mark_start"],
            "mark_end": top_mark["mark_end"],
            "in_tail": top_mark["mark_end"] >= buffer.shape[0] - mark_length_tolarence
        }

        # Find bottom most mark (maximum y position)
        bottom_mark = max(valid_marks, key=lambda m: m["mark_end"])
        ret_dict["bottom_mark"] = {
            "mark_exists": True,
            "mark_start": bottom_mark["mark_start"],
            "mark_end": bottom_mark["mark_end"],
            "in_tail": bottom_mark["mark_end"] >= buffer.shape[0] - mark_length_tolarence
        }

        # Find mark with area closest to threshold
        best_area_mark = min(valid_marks, key=lambda m: m["area_diff"])
        ret_dict["best_area_mark"] = {
            "mark_exists": True,
            "mark_start": best_area_mark["mark_start"],
            "mark_end": best_area_mark["mark_end"],
            "area_diff": best_area_mark["area_diff"],
            "in_tail": best_area_mark["mark_end"] >= buffer.shape[0] - mark_length_tolarence

        }
        return ret_dict

    def check_mark(self, block: numpy.ndarray, standard_mark_length: int, mark_roi: tuple, margin: int, mark_area_ratio_thresh: float, mark_roi_binary_thresh: int, mark_length_tolarence: int, ratio: float = 0.05) -> dict:
        """
        Analyzes a given region of interest (ROI) in an image block to detect the presence of a mark 
        and determine its start and end positions.
        Args:
            block (numpy.ndarray): The input image block to analyze.
            standard_mark_length (int): The standard length of the mark for comparison.
            mark_roi (tuple): A tuple (x, y, w, h) defining the region of interest in the image block.
            margin (int): The margin to extend above and below the ROI for analysis.
            mark_area_ratio_thresh (float): The threshold for the ratio of the mark's area to the ROI area.
            mark_roi_binary_thresh (int): The binary threshold value for binarizing the ROI.
            mark_length_tolarence (int): The tolerance for deviations in the mark's length.
            ratio (float, optional): Scaling ratio for resizing the ROI. Defaults to 0.05.
        Returns:
            dict: A dictionary containing:
                - "mark_exists" (bool): Whether a mark was detected.
                - "mark_start" (float): The starting position of the detected mark in the original coordinates.
                - "mark_end" (float): The ending position of the detected mark in the original coordinates.
        """

        ret_dict = {
            "mark_exists": False,
            "mark_start": -inf,
            "mark_end": inf,
            "in_tail": False
        }

        x, y, w, h = mark_roi
        start_x = x
        end_x = x + w
        start_y = y - margin
        end_y = y + h + margin
        roi = block[start_y:end_y, start_x:end_x]
        roi_width = roi.shape[1]

        if ratio != 1.0:
            scaled_w = int(roi.shape[1] * ratio)
            scaled_h = int(roi.shape[0] * ratio)
            _image = cv2.resize(roi, (scaled_w, scaled_h))
        else:
            _image = roi

        # adjust buffer_width and mark_length based on the scaling ratio
        scaled_roi_width = int(roi_width * ratio)
        scaled_standard_mark_length = int(standard_mark_length * ratio)

        cut_edge_coefficient = 0.3  # 剪切系数
        margin = round(cut_edge_coefficient * scaled_roi_width)
        margin = margin if margin > 0 else 1

        # concentrate on the center of the image
        roi = _image[:, margin:-margin]

        if len(roi.shape) == 3:
            if roi.shape[2] == 3:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(roi, mark_roi_binary_thresh,
                               255, cv2.THRESH_BINARY)[1]

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8, ltype=cv2.CV_32S)
        # Calculate area threshold
        mark_area_thresh = scaled_standard_mark_length * scaled_roi_width * \
            (1 - 2 * cut_edge_coefficient) * mark_area_ratio_thresh

        # Track all valid marks
        largest_mark = 0
        mark_end = -1
        mark_start = -1
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area > mark_area_thresh:
                x = stats[label, cv2.CC_STAT_LEFT]
                y = stats[label, cv2.CC_STAT_TOP]
                w = stats[label, cv2.CC_STAT_WIDTH]
                h = stats[label, cv2.CC_STAT_HEIGHT]

                # Convert back to original coordinates
                start = int(y / ratio)
                end = int((y + h) / ratio)

                if area > largest_mark:
                    largest_mark = area
                    mark_end = end
                    mark_start = start
        if largest_mark == 0:
            return ret_dict
        ret_dict["mark_exists"] = True
        ret_dict["mark_end"] = mark_end + start_y
        ret_dict["mark_start"] = mark_start + start_y
        ret_dict["in_tail"] = True if ret_dict["mark_end"] >= roi.shape[0] - \
            mark_length_tolarence else False
        return ret_dict
