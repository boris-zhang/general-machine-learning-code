#!/home/dmer/.pyenv/versions/env3/bin/python
# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------
File Name   : general_logging.py
Description : 日志记录通用模块
Author      : Zhang Zhiyong
Created at  : 2018/12/08
---------------------------------------------------------------------------
"""
import sys
import traceback
import logging


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"

def write_log(path, logtype, loginfo):
    if logtype == "debug":
        logging.basicConfig(filename=path, level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
        logging.debug(loginfo)
    elif logtype == "info":
        logging.basicConfig(filename=path, level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
        logging.info(loginfo)
    elif logtype == "warning":
        logging.basicConfig(filename=path, level=logging.WARNING, format=LOG_FORMAT, datefmt=DATE_FORMAT)
        logging.warning(loginfo)
    elif logtype == "error":
        logging.basicConfig(filename=path, level=logging.ERROR, format=LOG_FORMAT, datefmt=DATE_FORMAT)
        logging.error(loginfo)
    elif logtype == "critical":
        logging.basicConfig(filename=path, level=logging.CRITICAL, format=LOG_FORMAT, datefmt=DATE_FORMAT)
        logging.critical(loginfo)
    else:
        logging.critical("Your log type %s is not supported: debug, info, warning, error, critical!" % logtype)
