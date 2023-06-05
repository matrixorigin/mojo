import altair as alt
import pandas as pd
import motr
import pytest
from vega_datasets import local_data, data

@pytest.fixture(scope="module")
def motr_init():
    motr.enable('mysql+pymysql://dump:111@localhost:6001/mojo')

def test_simple_bar_char(motr_init):
    source = pd.DataFrame({
            'a': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
                'b': [28, 55, 43, 91, 81, 53, 19, 87, 52]
                })

    motr.from_pd(source, 'simple')
    chart = alt.Chart('motr://simple').mark_bar().encode(
               x='a:N',
               y='b:Q'
    )

    chart = motr.transform_chart(chart)
    chart
    chart.save("chart.png")
