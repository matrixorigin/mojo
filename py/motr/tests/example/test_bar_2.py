import altair as alt
import pandas as pd
import motr
import pytest

@pytest.fixture(scope="module")
def motr_init():
    motr.enable('mysql+pymysql://dump:111@localhost:6001/mojo')

def test_bar_mean(motr_init):
    source = motr.from_table('wheat', ['year', 'wheat']) 
    bars = alt.Chart(source.url()).mark_bar().encode(
        x="year:O",
        y="wheat:Q",
    )
    bars = motr.transform_chart(bars, topLevel = False)

    rule = alt.Chart(source.url()).mark_rule(color='red').encode(
        y='mean(wheat):Q'
    )
    rule = motr.transform_chart(rule, topLevel = False)

    (bars + rule).save('bar_rule_chart.png')
