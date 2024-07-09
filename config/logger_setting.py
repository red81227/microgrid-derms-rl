"""This file is used in setting config."""

# import relation package.
import os
import logging
import logging.config
from logging.handlers import TimedRotatingFileHandler
import datetime as dt
import re


# import project package.
from config.project_setting import service_config


def get_logger():
    """Get the custom logging"""
    log_format = '[%(processName)s][%(threadName)s]%(asctime)s [%(levelname)s][%(module)s][%(funcName)s]%(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    saved_log_path = os.path.join(service_config.log_file_path, service_config.log_file_name)
    if not os.path.exists(service_config.log_file_path):
        os.makedirs(service_config.log_file_path, exist_ok=True)
    log_filename = dt.datetime.now().strftime(saved_log_path)

    handler = TimedRotatingFileHandler(
        log_filename, when='midnight', encoding='utf-8')

    formatter = logging.Formatter(log_format, datefmt=date_format)
    handler.setFormatter(formatter)
    handler.suffix = '-%Y-%m-%d'
    handler.extMatch = re.compile(r"^\d{8}$")
    # logging.getLogger().addHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            handler,
            logging.StreamHandler()
        ]
    )
    # commente first because we need to look
    logging.getLogger('apscheduler.executors.default').propagate = False
    return logging


log = get_logger()
