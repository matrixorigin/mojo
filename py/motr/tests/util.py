import sys
import sqlalchemy
from sqlalchemy import create_engine
from vega_datasets import local_data, data

def import_motr():
    import os
    import sys
    import inspect
    currdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currdir)
    sys.path.insert(0, parentdir)
    import motr

def load_vega_datasets():
    e = create_engine('mysql+pymysql://dump:111@localhost:6001/mojo')
    datasets = local_data.list_datasets()
    for d in datasets:
        tbl = d.replace('-', '_')
        print("loading {0} into {1}".format(d, tbl))
        df = data(d)
        df.to_sql(tbl, e, if_exists='replace', index=False)

def load_data(name):
    e = create_engine('mysql+pymysql://dump:111@localhost:6001/mojo')
    tbl = name.replace('-', '_')
    print("loading {0} into {1}".format(name, tbl))
    df = data(name)
    df.to_sql(tbl, e, if_exists='replace', index=False)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        load_vega_datasets()
    else:
        load_data(sys.argv[1])
