import altair as alt
import pandas as pd
from . import modb

# MO connection object
__CONN__ = None
__MOCK__ = {}

def mock_view(name, data):
    __MOCK__[name] = data

def from_sql(sql, alias=""):
    return __CONN__.from_sql(sql, alias)

def build_query(qry, alias=""):
    return __CONN__.build_query(qry, alias)

def transform_chart(chart):
    return chart

def enable(host='localhost', port=6001, user='dump', password='111', database='mojo'):
    global __CONN__
    __CONN__ = modb.connect(host, port, user, password, database)

def disable():
    if __CONN__ is not None:
        __CONN__.close()
    __CONN__ = None
