# open a mysql connection

import pymysql

def connect(host, port, user, password, database):
    return Conn(host, port, user, password, database)

class Conn:
    def __init__(self, host, port, user, password, database):
        self.conn = pymysql.connect(host=host, port=port, user=user, password=password, database=database)
        self.cursor = self.conn.cursor()
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
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    def close(self):
        self.conn.close()

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
            return
        rsql = self.resolve_col(self.origsql)
        if self.inputs == None or len(self.inputs) == 0:
            self.sql = rsql
        else:
            self.sql = "WITH "
            self.sql += ",\n".join([xtn + " as (" + self.conn.getxt(xtn).sql + ")" for xtn in self.inputs])
            self.sql += "\n"
            self.sql += rsql
    
    def execute(self):
        self.build_sql()
        return self.conn.query(self.sql)
