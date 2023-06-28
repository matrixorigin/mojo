import altair as alt
import numpy as np
import pandas as pd
import motr
import pytest
from vega_datasets import local_data, data


@pytest.fixture(scope="module")
def motr_init():
    motr.enable('mysql+pymysql://dump:111@localhost:6001/mojo')


def test_simple_bar_chart(motr_init):
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
    chart.save("simple_bar_chart.png")


def test_simple_heatmap(motr_init):
    # Compute x^2 + y^2 across a 2D grid
    x, y = np.meshgrid(range(-5, 5), range(-5, 5))
    z = x ** 2 + y ** 2

    # Convert this grid to columnar data expected by Altair
    source = pd.DataFrame({'x': x.ravel(),
                           'y': y.ravel(),
                           'z': z.ravel()})

    motr.from_pd(source, 'simple')
    chart = alt.Chart("motr://simple").mark_rect().encode(
        x='x:O',
        y='y:O',
        color='z:Q'
    )
    chart = motr.transform_chart(chart)
    chart.save("simple_heatmap_chart.png")
