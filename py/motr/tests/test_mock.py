import altair as alt
import numpy as np
import pandas as pd

# not necessary, if running from parent dir using pytest.
# from .util import import_motr
# import_motr()
import motr

def test_mock():
    # enable motr
    motr.enable()

    rand = np.random.RandomState(12345)

    df = pd.DataFrame({
        'x': np.arange(200),
        'y': rand.randn(200).cumsum()
    })

    # create a mock view
    motr.mock_view('mock1', df)
    mockurl = "motr://mock1"

    points = alt.Chart(mockurl).mark_point().encode(
        x='x:Q',
        y='y:Q'
    )

    line = alt.Chart(mockurl).transform_window(
        ymean='mean(y)',
        sort=[alt.SortField('x')],
        frame=[5, 5]
    ).mark_line(color='red').encode(
        x='x:Q',
        y='ymean:Q'
    )

    # why is my data_transformer not working?
    chart = points + line
    # nyi, noop now.
    chart = motr.transform_chart(chart)
    
    # NYI 
    # chart.save("chart.png")
