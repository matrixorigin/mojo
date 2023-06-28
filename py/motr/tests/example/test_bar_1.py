import altair as alt
import pandas as pd
import motr
import pytest


@pytest.fixture(scope="module")
def motr_init():
    motr.enable('mysql+pymysql://dump:111@localhost:6001/mojo')


def test_bar_highlight(motr_init):
    source = motr.from_table('wheat', ['year', 'wheat'])
    threshold = pd.DataFrame([{"threshold": 90}])
    bars = alt.Chart(source.url()).mark_bar().encode(
        x="year:O",
        y="wheat:Q",
    )
    bars = motr.transform_chart(bars, topLevel=False)
    print(bars.to_json())

    highlight = alt.Chart(source.url()).mark_bar(color="#e45755").encode(
        x='year:O',
        y='baseline:Q',
        y2='wheat:Q'
    ).transform_filter(
        alt.datum.wheat > 90
    ).transform_calculate("baseline", "90")
    highlight = motr.transform_chart(highlight, topLevel=False)
    print(highlight.to_json())

    rule = alt.Chart(threshold).mark_rule().encode(
        y='threshold:Q'
    )
    print(rule.to_json())

    three = bars + highlight + rule
    three.properties(width=600)
    print(three.to_json())
    three.save('bar_highlight_chart.png')


def test_bar_highlight_orig():
    import altair as alt
    import pandas as pd
    from vega_datasets import data

    source = data.wheat()
    threshold = pd.DataFrame([{"threshold": 90}])

    bars = alt.Chart(source).mark_bar().encode(
        x="year:O",
        y="wheat:Q",
    )

    highlight = alt.Chart(source).mark_bar(color="#e45755").encode(
        x='year:O',
        y='baseline:Q',
        y2='wheat:Q'
    ).transform_filter(
        alt.datum.wheat > 90
    ).transform_calculate("baseline", "90")

    rule = alt.Chart(threshold).mark_rule().encode(
        y='threshold:Q'
    )

    (bars + highlight + rule).properties(width=600)
