import motr

def test_iris():
    motr.enable()

    iris = motr.from_sql("select * from iris", "tmpt")
    iris_cnt = motr.build_query("select count(*) from @tmpt@")
    cnt = iris_cnt.execute()

    assert cnt[0][0] == 150