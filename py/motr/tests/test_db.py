import motr
import pytest
from vega_datasets import local_data, data

@pytest.fixture(scope="module")
def motr_init():
    motr.enable('mysql+pymysql://dump:111@localhost:6001/mojo')

def test_iris(motr_init):
    iris = motr.from_sql("select * from iris", "tmpt")
    iris_cnt = motr.build_query("select count(*) c from @tmpt@")
    cnt = iris_cnt.execute()
    assert cnt.at[0,'c'] == 150

def test_iris_pd(motr_init):
    iris = data('iris')
    iris = motr.from_pd(iris, "iris")
    iris_cnt = motr.build_query("select count(*) c from @iris@")
    cnt = iris_cnt.execute()
    assert cnt.at[0,'c'] == 150
