import altair as alt
from altair_transform import transform_chart
import pandas as pd
import numpy as np
import pytest

def test_simple_hist_spec():
    np.random.seed(12345)
    df = pd.DataFrame({
        'y': np.random.rand(20)
        })
    chart = alt.Chart(df).mark_bar().encode(
            alt.Y("y", bin=True),
            x='count()',
    )
    print(chart.to_json())

    chart2 = transform_chart(chart)
    print(chart2.to_json())

if __name__ == '__main__':
    test_simple_hist_spec()
