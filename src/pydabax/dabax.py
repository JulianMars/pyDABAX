# -*- coding: utf-8 -*-
"""
Access to dabax in TinyDB (json) format.
"""

import json
import logging
from pathlib import Path

from pandas import DataFrame
from tinydb import TinyDB, Query
from tinydb.storages import MemoryStorage

__author__ = "Julian Mars"
__copyright__ = "Julian Mars"
__license__ = "mit"

_logger = logging.getLogger(__name__)

latest_json = Path(__file__).parent / "data/xraydb.latest.json"


class Dabax:
    def __init__(self, path_to_db):
        self.dbb = TinyDB(storage=MemoryStorage)
        with open(path_to_db, 'r') as json_file:
            data = json.load(json_file)
            self.dbb.storage.write(data)
            self.db = self.dbb.table("Elements")

    def read_db(self, symbol):
        entry = Query()
        # noinspection PyTypeChecker
        return self.db.search(entry.symbol == symbol)[0]




    def get_entry(self, symbol, keys):
        res = self.read_db(symbol)
        if not isinstance(keys, list):
            return res[keys]
        for key in keys[:-1]:  # go down
            res = res[key]
        ans = res[keys[-1]]  # access last key in chain
        return ans

    def get_keys(self, symbol, keys=False):
        res = self.read_db(symbol)
        if not keys:
            return list(res.keys())
        if not isinstance(keys, list):
            return list(res[keys].keys())
        for key in keys[:-1]:  # go down
            res = res[key]
        ans = list(res[keys[-1]].keys())  # access last key in chain)
        return ans

    def get_table(self, symbol, table_name):
        res = self.read_db(symbol)
        df = DataFrame.from_dict(res[table_name]["table"])
        return df

    def get(self, symbol, keys):
        try:
            ans = self.get_table(symbol, keys)
        except TypeError:
            ans = self.get_entry(symbol, keys)
        return ans

print("Loading database from " + str(latest_json))
dabax = Dabax(latest_json)
