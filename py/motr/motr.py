import altair as alt
import pandas as pd
from . import modb

# MO connection object
__CONN__ = None

def from_sql(sql, alias=""):
    return __CONN__.from_sql(sql, alias)

def build_query(qry, alias=""):
    return __CONN__.build_query(qry, alias)

def transform_chart(chart):
    return chart

def enable(connstr):
    global __CONN__
    __CONN__ = modb.Conn(connstr)

def disable():
    if __CONN__ is not None:
        __CONN__.close()
    __CONN__ = None
