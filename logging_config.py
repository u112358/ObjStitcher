from datetime import datetime
import logging
import os
from colorlog import ColoredFormatter


class SingletonLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.setup_logger()
        return cls._instance

    def setup_logger(self):
        self.logger = logging.getLogger('ObjStitcher')
        self.logger.setLevel(logging.DEBUG)

        # 控制台处理器（带颜色）
        console_handler = logging.StreamHandler()
        console_formatter = ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(console_formatter)

        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_filename = os.path.join(
            log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # 添加处理器
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)


# 全局日志记录器
logger = SingletonLogger().logger
logger.info("Logger initialized.")
