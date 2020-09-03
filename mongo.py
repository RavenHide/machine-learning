# -*- coding: utf-8 -*-
"""
    mongo
    ~~~~~~~

    Description.

    :author: zhaoyouqing
    :copyright: (c) 2018, Tungee
    :date created: 2020-06-15
    :python version: 3.5
"""


from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.read_preferences import ReadPreference


class MongoDBClientWrapper(object):
    def __init__(self, mongo_url, database, user, password, max_pool_size=50):
        self._mongo_url = mongo_url
        self._database = database
        self._user = user
        self._password = password

        self._client = MongoClient(
            self._mongo_url, connect=False, maxPoolSize=max_pool_size,
        )
        self.db = self._client[self._database]

        if self._user is not None and self._password is not None:
            self.db.authenticate(self._user, self._password)

        self._p_col_map = dict()
        self._s_col_map = dict()

    def p_col(self, col_name):
        if col_name not in self._p_col_map:
            self._p_col_map[col_name] = Collection(
                self.db, col_name,
                read_preference=ReadPreference.PRIMARY_PREFERRED
            )
        return self._p_col_map[col_name]

    def s_col(self, col_name):
        if col_name not in self._s_col_map:
            self._s_col_map[col_name] = Collection(
                self.db, col_name,
                read_preference=ReadPreference.SECONDARY_PREFERRED
            )
        return self._s_col_map[col_name]

    def __del__(self):
        try:
            self._client.close()
        except:
            pass