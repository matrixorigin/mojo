import altair as alt
import numpy as np
import pandas as pd

rand = np.random.RandomState(12345)

df = pd.DataFrame({
    'x': np.arange(200),
    'y': rand.randn(200).cumsum()
})

points = alt.Chart(df).mark_point().encode(
    x='x:Q',
    y='y:Q'
)

line = alt.Chart(df).transform_window(
    ymean='mean(y)',
    sort=[alt.SortField('x')],
    frame=[5, 5]
).mark_line(color='red').encode(
    x='x:Q',
    y='ymean:Q'
)

# print(line.to_json())

chart = points + line
chart

# chart.save("chart.png")

