import pdb
import traceback
from logging_config import logger


class ResultMerger():

    def __init__(self):
        logger.info(
            "to work with timesai, instance has to be initialised manually")
        self.initialised = False
        """
        frame_list = [1, 2, 3, 4]
        frame_offsets = [[0, 4999], [5000, 9999], [10000, 14999], [15000, 19999]]
        defects_list = [{},{},{},{}], 
        """

    def _init_timesai_tool(self, overlap):
        self.defects_list = []
        self.overlap = overlap

        self.initialised = True

    def merge_results(self, frame_list, frame_offsets):
        merged_results = [[]]
        merged_labels = [[]]
        for i, _frame_id in enumerate(frame_list):
            defect = self.defects_list[i]
            frame_rois = defect['frame_rois']
            frame_defect_labels = defect['frame_defect_labels']
            for roi_idx, roi in enumerate(frame_rois[0]):
                x = roi[1]
                y = roi[2] + frame_offsets[i][0]
                w = roi[3]
                h = roi[4]
                reversed_roi = [roi[0], x, y, w, h, roi[5]]
                if not (y < self.overlap
                        or y + h > self.obj_size - self.overlap):
                    merged_results[0].append(reversed_roi)
                    merged_labels[0].append(frame_defect_labels[roi_idx])

        if frame_offsets[-1][1] > self.obj_size:
            self.defects_list = self.defects_list[-1:]
        else:
            self.defects_list = []

        return merged_results, merged_labels

    def merge_results_timesai_tool(self, *args):
        ret_dict = {'0': 1, '1': "error", '2': [], '3': [], '4': 999}
        try:

            args = args[0]
            frame_list = args[0]
            frame_offsets = args[1]
            frame_id = args[2]
            frame_roi = args[3]
            obj = args[4][0]  # 输入大图用于显示，并且要判断defect_list哪里需要保留
            if obj is not None:  # 不是每次都有图输入
                self.obj_size = obj.shape[0]
            print("args[5]=",args[5],flush=True)
            frame_defect_labels = args[5][0]
            print("frame_defect_labels=", frame_defect_labels, flush=True)

            overlap = args[6]

            if args[7] is not None:
                reset_signal = args[7]
            else:
                reset_signal = 0

            if not self.initialised:
                self._init_timesai_tool(overlap)
            if reset_signal == 5:
                self._init_timesai_tool(overlap)
                ret_dict['4'] = 0

            self.defects_list.append({
                'id': frame_id,
                'frame_rois': frame_roi,
                'frame_defect_labels': frame_defect_labels
            })
            # 图还没拼完
            if len(frame_list) == 4 and ('WAIT' in (chr(frame_list[0]) + chr(
                    frame_list[1]) + chr(frame_list[2]) + chr(frame_list[3]))):

                ret_dict['0'] = 0
                ret_dict['1'] = 'not enough results'
            else:
                merged_rects, merged_labels = self.merge_results(
                    frame_list, frame_offsets)

                ret_dict['0'] = 0
                ret_dict['1'] = 'done'
                ret_dict['2'] = merged_rects
                ret_dict['3'] = merged_labels
        except Exception as e:
            traceback.print_exc()
            ret_dict['1'] = str(e)
        return ret_dict
