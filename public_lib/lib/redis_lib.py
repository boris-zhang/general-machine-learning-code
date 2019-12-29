#!/home/dmer/.pyenv/versions/env4/bin/python
# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------------
File Name   : redis_publib.py
Description : redis公共库
Author      : Zhangzhiyong
Created at  : 2018/09/19
--------------------------------------------------------------------------
"""
import sys
import numpy as np
import pandas as pd
from rediscluster import StrictRedisCluster

def redis_connect():
    redisNodes = [{'host':'172.21.0.25','port':7000},
                  {'host':'172.21.0.25','port':7001},
                  {'host':'172.21.0.25','port':7002},
                  {'host':'172.21.0.27','port':7000},
                  {'host':'172.21.0.27','port':7001},
                  {'host':'172.21.0.27','port':7002}]
    try:
        rs = StrictRedisCluster(startup_nodes=redisNodes, password='Cdg3fD34fgrDrE3Gsio089hrDEer7gedEs4x')
    except Exception:
        print('failed to connect redis cluster!')
        sys.exit(1)

    return rs