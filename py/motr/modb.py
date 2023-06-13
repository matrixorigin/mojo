# open a mysql connection

from sqlalchemy import create_engine
import pandas

class Conn:
    def __init__(self, connstr): 
        self.eng = create_engine(connstr)
        self.nexttmp = 0
        self.xts = {}

    def rmxt(self, x):
        if x in self.xts:
            del self.xts[x]

    def getxt(self, x):
        if x in self.xts:
            return self.xts[x]
        return None

    def next_tmpname(self):
        self.nexttmp += 1 
        return "tmp_{0}".format(self.nexttmp)

    def query(self, sql):   
        return pandas.read_sql(sql, self.eng)

    def close(self):
        self.eng.dispose()

    def build_query(self, qry, alias=""):
        xt = Table(self, qry, alias)
        xt.build_sql()
        self.xts[xt.alias] = xt
        return xt

    def from_sql(self, qry, alias=""):
        xt = Table(self, qry, alias)
        xt.sql = qry 
        self.xts[xt.alias] = xt
        return xt
    
    def from_table(self, t, alias=""):
        if alias == "":
            alias = t
        return self.from_sql("select * from " + t, alias)

    def from_pd(self, df, alias=""):
        if alias == "":
            alias = self.next_tmpname()
        df.to_sql("motr_tmp_" + alias, self.eng, if_exists='replace', index=False)
        return self.from_table("motr_tmp_" + alias, alias)

class Table:
    def __init__(self, conn, origsql="", alias=""):
        self.conn = conn
        self.origsql = origsql
        if alias == "":
            alias = self.conn.next_tmpname()
        self.alias = alias
        self.sql = None
        self.inputs = {}

    # @x.y@ where x is a table alias, y is colname -> tablealias.colname
    # @@ will print out a single @
    def resolve_col(self, s):
        strs = s.split('@')
        rs = []
        i = 0
        while i < len(strs):
            rs.append(strs[i])
            i += 1 

            if i == len(strs):
                break

            if strs[i] == '':
                rs.append('@')
            else:
                # record alias table.
                xy = strs[i].split('.')
                self.inputs[xy[0]] = 1
                rs.append(strs[i])
            i += 1 
        # print(rs)
        return "".join(rs) 

    # build a valid sql query string
    def build_sql(self): 
        if self.sql != None:
            return self.sql
        rsql = self.resolve_col(self.origsql)
        if self.inputs == None or len(self.inputs) == 0:
            self.sql = rsql
        else:
            self.sql = "WITH "
            self.sql += ",\n".join([xtn + " as (" + self.conn.getxt(xtn).sql + ")" for xtn in self.inputs])
            self.sql += "\n"
            self.sql += rsql
        return self.sql

    def execute(self):
        self.build_sql()
        return self.conn.query(self.sql)

    def transform_bin(self, opts):
        # This is the number of bins version. 
        # TODO: add step size version
        if 'step' in opts:
            return self, False

        # decide number of bins
        if 'maxbins' in opts:
            numbins = opts['maxbins']
        else:
            numbins = 10

        selfsql = self.build_sql()
        
        sql = f"""
            WITH 
            minmax AS ( select min({opts['field']}) as minval, max({opts['field']}) as maxval, 
                        (max({opts['field']}) - min({opts['field']})) / {numbins} as binwidth 
                        from ({selfsql}) {self.alias} ),
            binned AS ( select case when minmax.binwidth = 0 then 0
                                    else floor(({opts['field']} - minmax.minval) / minmax.binwidth) end as {opts['field']}_binnumber, 
                               minmax.minval + case when minmax.binwidth = 0 then 0 
                                    else minmax.binwidth * floor(({opts['field']} - minmax.minval) / minmax.binwidth) 
                                    end as {opts['field']}_binned,
                               minmax.minval + minmax.binwidth + case when minmax.binwidth = 0 then 0 
                                    else minmax.binwidth * floor(({opts['field']} - minmax.minval) / minmax.binwidth) 
                                    end as {opts['field']}_binned2,
                                {self.alias}.* 
                        from ({selfsql}) {self.alias}, minmax )
            """

        if 'aggregate' not in opts:
            sql += "select * from binned"
        else:
            sql += f""" select any_value({opts['field']}_binned) as {opts['field']}_binned, 
                            any_value({opts['field']}_binned2) as {opts['field']}_binned2 """
            for agg in opts['aggregate']:
                if 'field' not in agg:
                    aggcol = "1"
                    aggcolAs = f"__{agg['aggregate']}"
                else:
                    aggcol = agg['field']
                    aggcolAs = f"__{agg['aggregate']}_{agg['field']}"
                sql += f""", {agg['aggregate']}({aggcol}) as {aggcolAs} """
            sql += f""" from binned group by {opts['field']}_binnumber """

        # debug
        # print(sql)
        return self.conn.from_sql(sql, self.alias + "_binned"), True