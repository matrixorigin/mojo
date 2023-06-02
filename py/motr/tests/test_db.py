import motr

def test_iris():
    motr.enable('mysql+pymysql://dump:111@localhost:6001/mojo')
    iris = motr.from_sql("select * from iris", "tmpt")
    iris_cnt = motr.build_query("select count(*) c from @tmpt@")
    cnt = iris_cnt.execute()
    assert cnt.at[0,'c'] == 150
