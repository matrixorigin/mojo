import altair as alt

class Transformer:
    def __init__(self, conn):
        self.conn = conn

    def extract_encoding_transform(self, chart):
        """ Extract transform from encodings """
        if 'encoding' not in chart:
            return chart
        # NYI
        return chart

    def do_transform(self, chart):
        chart = self.extract_encoding_transform(chart)
        return chart

    # mo_exec will run the xtable query and replace data with result
    def mo_exec(self, chart):
        # only execute motr://
        if 'data' not in chart or 'url' not in chart['data']:
            return chart

        url = chart['data']['url']
        if not url.startswith("motr://"):
            return chart

        # x is the xtable name
        x = url[7:]
        xt = self.conn.getxt(x)
        pd = xt.execute()
        del chart['data']['url']
        chart['data']['values'] = pd.to_dict(orient='records')
        return chart

    # following altir_transform tranform_chart
    def transform(self, chart):
        tr = chart.to_dict()
        tr = self.do_transform(tr)
        tr = self.mo_exec(tr)
        return alt.Chart.from_dict(tr)