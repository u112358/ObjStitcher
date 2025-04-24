from datetime import datetime
import logging
import os


class SingletonLogger:
    _instance = None
    _loggers = {}  # 存储各实例的logger

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.setup_base_logger()
        return cls._instance

    def setup_base_logger(self):
        """基础日志配置"""
        self.base_logger = logging.getLogger('ObjStitcher')
        self.base_logger.setLevel(logging.DEBUG)

        # 清除现有handler
        for handler in self.base_logger.handlers[:]:
            self.base_logger.removeHandler(handler)
            handler.close()

    def get_instance_logger(self, instance_name):
        """
        获取实例专属logger
        :param instance_name: 实例标识名
        :return: 配置好的日志记录器
        """
        # 如果已有logger，先清理旧handler
        logger = self._loggers.get(instance_name)
        if logger:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()
        else:
            logger = logging.getLogger(f'ObjStitcher.{instance_name}')
            logger.setLevel(logging.DEBUG)
            logger.propagate = False  # 禁止传播到父logger
            self._loggers[instance_name] = logger

        # 统一日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )

        # 控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 文件handler
        log_dir = 'obj_stitcher_logs'
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(
                log_dir, f"{instance_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            print(f"log file has been created: {os.path.abspath(log_file)}")
        except Exception as e:
            print(f"create log file failed: {str(e)}")

        return logger
