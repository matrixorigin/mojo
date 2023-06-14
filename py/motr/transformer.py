import altair as alt

from .modb import gen_aggcol_as

class Transformer:
    def __init__(self, conn):
        self.conn = conn

    def do_one_transform(self, chart, data, transform):
        xt = self.data_to_xt(data)
        if xt is None:
            return chart, data, False

        newdata, ok = None, False
        if 'bin' in transform:
            newxt, ok = xt.transform_bin(**transform)
        elif 'aggregate' in transform:
            newxt, ok = xt.transform_aggregate(**transform)
        else:
            # NYI
            pass
        
        if not ok:
            return chart, data, False
        else:
            newdata = newxt.urldata()
            chart['data'] = newdata
            return chart, newdata, True

    def transform_aggregate_encoding(self, chart, data, encoding):
        newencoding = {}
        aggops = {'aggregate': [], 'groupby': []}
        for k in encoding:
            ek = encoding[k]
            if 'aggregate' in ek:
                _, _, aggcolAs = gen_aggcol_as(ek)
                aggops['aggregate'].append(ek)
                newencoding[k] = ek.copy()
                del newencoding[k]['aggregate']
                if 'field' in ek:
                    newencoding[k]['field'] = aggcolAs
                    newencoding[k]['title'] = aggcolAs + ' of ' + ek['field']
                else:
                    newencoding[k]['field'] = aggcolAs
                    newencoding[k]['title'] = ek['aggregate'] + ' of Records'
            else:
                if 'field' in ek:
                    aggops['groupby'].append(ek['field'])
                newencoding[k] = ek.copy()

        newchart, newdata, ok = self.do_one_transform(chart, data, aggops)
        if ok:
            newchart['encoding'] = newencoding
            return newchart, newdata
        else:
            return chart, data

    # extract bin transform from encoding
    def transform_bin_encoding(self, chart, data, encoding):
        binop = {}
        aggops = {'aggregate': [], 'groupby': []} 
        newencoding = {}

        for k in encoding:
            ek = encoding[k]
            if 'bin' in ek and ek['bin'] == True:
                binop['bin'] = ek['bin']
                binop['field'] = ek['field']
                for opt in 'as', 'anchor', 'base', 'binned', 'devide', 'extent', 'maxbins', 'minstep', 'nice', 'step', 'steps':
                    if opt in ek: 
                        binop[opt] = ek[opt] 

                # redirect newencoding to columns after transform.
                # make a copy so that if transform failed we do not mess up
                # old encoding.
                binned = ek['field'] + '_binned'
                binned2 = ek['field'] + '_binned2'
                newencoding[k] = ek.copy()
                newencoding[k]['bin'] = 'binned'
                newencoding[k]['field'] = binned
                if 'title' in ek:
                    newencoding[k]['title'] = ek['title']
                else:
                    newencoding[k]['title'] = ek['field'] + ' (binned)'
                if k == 'x' or k == 'y':
                    newencoding[k + '2'] = {'field': binned2} 
                # add groupby
                aggops['groupby'].append(binned)
                aggops['groupby'].append(binned2)

            # transform aggregate as well
            if 'aggregate' in ek:
                aggop = {}
                for opt in ['aggregate', 'field', 'as', 'groupby']: 
                    if opt in ek:
                        aggop[opt] = ek[opt]
                aggops['aggregate'].append(aggop) 

                # redirect agg to columns after transform.
                newencoding[k] = ek.copy()
                del newencoding[k]['aggregate']
                if 'field' in ek:
                    newencoding[k]['field'] = '__' + ek['aggregate'] + '_' + ek['field']
                    newencoding[k]['title'] = ek['aggregate'] + ' of ' + ek['field']
                else:
                    newencoding[k]['field'] = '__' + ek['aggregate']
                    newencoding[k]['title'] = ek['aggregate'] + ' of Records'

        if 'bin' not in binop:
            return chart, data
        else:
            newchart, newdata, ok = self.do_one_transform(chart, data, binop)
            if ok and len(aggops['aggregate']) > 0:
                newchart, newdata, ok = self.do_one_transform(newchart, newdata, aggops) 

            if ok:
                newchart['encoding'] = newencoding
                return newchart, newdata
            else:
                # transform failed, return original chart
                return chart, data

    def transform_encoding(self, chart, data):
        """ Extract transform from encodings """
        if 'encoding' not in chart:
            return chart, data, None

        # A list of known tranformas
        trs = {'bin': 0, 
               'aggregate': 0, 
               'other': 0
            }
        
        for k in chart['encoding']:
            if 'bin' in chart['encoding'][k]:
                trs['bin'] += 1
            elif 'aggregate' in chart['encoding'][k]:
                trs['aggregate'] += 1
            else:
                trs['other'] += 1
            
        newtr, newencoding = None, None
        if trs['bin'] == 1:
            # Handle first case, bin.   We can bin only if all or none is aggregate. 
            if trs['aggregate'] == 0 or trs['other'] == 0:
                newchart, newdata = self.transform_bin_encoding(chart, data, chart['encoding'])
                return newchart, newdata
        elif trs['bin'] == 0 and trs['aggregate'] > 0:
            # Handle aggregate
            newchart, newdata = self.transform_aggregate_encoding(chart, data, chart['encoding'])
            return newchart, newdata

        # did not extract any transform
        return chart, data

    def do_transform(self, chart):
        # only execute motr://
        if 'data' not in chart or 'url' not in chart['data']:
            return chart
        data = chart['data']

        if 'transform' in chart:
            transforms = chart['transform']
            remaining = []
            for i in range(len(transforms)):
                chart, data, ok = self.do_one_transform(chart, data, transforms[i])
                if not ok:
                    remaining = transforms[i:]
                    break

            if len(remaining) == 0:
                del chart['transform']
            else:
                chart['transform'] = remaining

        # if we can perform all transforms, try to transform encoding
        if 'transform' not in chart:
            newchart, _ = self.transform_encoding(chart, data)
            return newchart
        else:
            return chart

    # mo_exec will run the xtable query and replace data with result
    def mo_exec(self, chart):
        # only execute motr://
        if 'data' not in chart or 'url' not in chart['data']:
            return chart

        xt = self.data_to_xt(chart['data'])
        if xt is None:
            return chart

        pd = xt.execute()
        del chart['data']['url']
        chart['data']['values'] = pd.to_dict(orient='records')
        return chart

    def data_to_xt(self, data):
        url = data['url']
        if not url.startswith("motr://"):
            return None
        # x is the xtable name
        x = url[7:]
        xt = self.conn.getxt(x)
        return xt


    # following altir_transform tranform_chart
    def transform(self, chart, topLevel=True): 
        # test if chart.config is undefined
        tr = chart.to_dict()
        tr = self.do_transform(tr)
        tr = self.mo_exec(tr)
        if not topLevel:
            for topKey in ['background', 'padding', 'autosize', 'config', '$schema']:
                if topKey in tr:
                    del tr[topKey]
        return alt.Chart.from_dict(tr)
