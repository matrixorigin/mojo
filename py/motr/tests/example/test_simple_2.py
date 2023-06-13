import altair as alt
import numpy as np
import pandas as pd
import motr
import pytest

@pytest.fixture(scope="module")
def motr_init():
    motr.enable('mysql+pymysql://dump:111@localhost:6001/mojo')

def test_simple_hist(motr_init):
    # Note that altair is case sensitive, alias the columns.
    motr.from_sql('select imdb_rating as IMDB_Rating from movies', 'simple')
    chart = alt.Chart('motr://simple').mark_bar().encode(
            alt.X("IMDB_Rating:Q", bin=True),
            y='count()',
    )
    chart = motr.transform_chart(chart)
    print(chart.to_json())
    chart.save("simple_hist_chart.png")

if __name__ == "__main__":
    pytest.main([__file__])
