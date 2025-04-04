import os
import cv2
from logging_config import logger


class FrameCapturer(object):
    """
    FrameCapturer class for capturing frames from a local directory.
    Attributes:
        file_path (str): The path to the local file directory.
        file_list (list): List of image files in the directory.
        file_idx (int): Index of the current file being processed.
        total_files (int): Total number of image files in the directory.
        visiable (bool): A flag to set visibility of the frames.
    Methods:
        ready_to_capture():
        next_frame():
        _get_next_frame():
        Internal method to read the next frame from the directory.
    """

    def __init__(self, mode='local', file_path=None, visiable=False):
        """
        Initializes the FrameCapturer object.
        Parameters:
        mode (str): The mode of frame capturing. It can be 'camera' or 'local'. Default is 'camera'.
        file_path (str, optional): The path to the local file if mode is 'local'. Default is None.
        visiable (bool): A flag to set visibility. Default is False.
        Raises:
        NotImplementedError: If mode is 'camera' as capturing frames from camera is not implemented yet.
        ValueError: If mode is 'local' and file_path is not provided.
        ValueError: If mode is neither 'camera' nor 'local'.
        """

        if mode == 'camera':
            raise NotImplementedError(
                "Capturing frames from camera is not implemented yet")
        elif mode == 'local':
            if not file_path:
                raise ValueError('file path is needed for local mode')
            else:
                self.file_path = file_path
                if not os.path.exists(self.file_path):
                    raise FileNotFoundError(
                        f"Directory {self.file_path} not found")
                self.file_list = [f for f in os.listdir(
                    self.file_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                if not self.file_list:
                    raise FileNotFoundError(
                        f"No files found in {self.file_path}")
                elif len(self.file_list) == 0:
                    raise FileNotFoundError(
                        f"No images found in {self.file_path}")
                self.current_file_idx = -1
                self.current_file_name = None
                self.total_files = len(self.file_list)

        else:
            raise ValueError(
                'mode should be camera or local file (jpg, png, jpeg supported)')
        self.visiable = visiable
        self.mode = mode

    def ready_to_capture(self):
        """
        Check if the system is ready to capture frames.
        Returns:
            bool: True if the system is in 'local' mode and the current file index is less than the total number of files, False otherwise.
        """

        return self.current_file_idx + 1 < self.total_files if self.mode == 'local' else False

    def next_frame(self):
        """
        Capture the next frame based on the current mode.
        In 'local' mode, it retrieves the next frame from a local source.
        In 'camera' mode, it raises a NotImplementedError as capturing from a camera is not yet implemented.
        Raises:
            IndexError: If there are no more frames to capture in 'local' mode.
            NotImplementedError: If the mode is set to 'camera'.
        Returns:
            frame: The next frame in 'local' mode.
        """

        if self.mode == 'local':
            if self.current_file_idx + 1 >= self.total_files:
                raise IndexError("No more frames to capture")
            self.current_file_idx += 1
            self.current_file_name = self.file_list[self.current_file_idx]
            frame = self._get_next_frame()
            return frame
        elif self.mode == 'camera':
            raise NotImplementedError(
                "Capturing frames from camera is not implemented yet")

    def _get_next_frame(self):
        """
        Retrieves the next frame based on the current mode.

        In 'local' mode, reads the next image file from the specified file path and displays it if visibility is enabled.
        In 'camera' mode, raises a NotImplementedError as capturing frames from the camera is not yet implemented.

        Returns:
            numpy.ndarray: The next frame as an image array in 'local' mode.

        Raises:
            NotImplementedError: If the mode is 'camera'.
        """

        if self.mode == 'local':
            frame = cv2.imread(os.path.join(
                self.file_path, self.file_list[self.current_file_idx]))
            if self.visiable:
                cv2.imshow('frame', frame)
                logger.warning(
                    'press any key to continue, if you do not want to see the frame, please initial the FrameCapturer with visiable=False')
                cv2.waitKey()
                cv2.destroyAllWindows
            return frame
        elif self.mode == 'camera':
            raise NotImplementedError(
                "Capturing frames from camera is not implemented yet")
