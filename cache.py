# -*- coding: utf-8 -*-
"""
    cache
    ~~~~~~~

    Description.

    :author: zhaoyouqing
    :copyright: (c) 2018, Tungee
    :date created: 2020-06-16
    :python version: 3.5
"""
import redis


class RedisConf(object):
    """Config of Redis"""
    HOST = 'localhost'
    PORT = 6379
    DB = 0
    PASSWORD = ''
    URL = 'redis://localhost:6379/2'
    CACHE_TTL = 3600


redis_inst = redis.StrictRedis(
    host=RedisConf.HOST,
    port=RedisConf.PORT,
    password=RedisConf.PASSWORD,
    db=RedisConf.DB
)