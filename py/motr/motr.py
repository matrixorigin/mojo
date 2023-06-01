import altair as alt
import pandas as pd

# MO connection object
__CONN__ = None
__MOCK__ = {}

def mock_view(name, data):
    __MOCK__[name] = data

def transform_chart(chart):
    return chart

def enable():
    pass

def disable():
    pass
