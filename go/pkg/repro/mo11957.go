package repro

import (
	"flag"
	"fmt"
	"sync"

	"github.com/abiosoft/ishell/v2"
	"github.com/matrixorigin/mojo/pkg/mo"
)

func Mo11957(sh *ishell.Context) {
	fs := flag.NewFlagSet("mo11957", flag.ContinueOnError)
	var thCnt int
	var loopCnt int
	var shareT bool
	fs.IntVar(&thCnt, "t", 10, "number of threads")
	fs.IntVar(&loopCnt, "n", 200, "loop count")
	fs.BoolVar(&shareT, "s", true, "if share table")

	var err error

	if err = fs.Parse(sh.Args); err != nil {
		sh.Println()
		return
	}
	sh.Printf("MO-11957: thread count:%d, loop:%d, shareT:%v\n", thCnt, loopCnt, shareT)

	modb := mo.DefaultDB()
	modb.Exec("create database if not exists repro")
	db, err := mo.OpenDB("6001", "repro")
	if err != nil {
		panic(err)
	}

	dbExec(db, "drop table if exists mo11957_10000")
	dbExec(db, "create table mo11957_10000(i int not null primary key auto_increment, j int, k int, t text)")
	dbExec(db, "insert into mo11957_10000 values(null, 1, 2, 'foo')")
	dbExec(db, "insert into mo11957_10000 values(null, 1, 2, 'bar')")

	var wg sync.WaitGroup
	wg.Add(thCnt)

	for i := 0; i < thCnt; i++ {
		go func(ii int, wg *sync.WaitGroup) {
			defer wg.Done()
			sh.Println("start worker", ii, "loop", loopCnt)

			dbExec(db, fmt.Sprintf("drop table if exists mo11957_%d", ii))
			dbExec(db, fmt.Sprintf("create table mo11957_%d(i int not null primary key auto_increment, j int, k int, t text)", ii))
			dbExec(db, fmt.Sprintf("insert into mo11957_%d values(null, 1, 2, 'foo')", ii))
			dbExec(db, fmt.Sprintf("insert into mo11957_%d values(null, 1, 2, 'bar')", ii))

			tx, err := db.Begin()
			if err != nil {
				panic(err)
			}

			tbn := 10000
			if !shareT {
				tbn = ii
			}

			for j := 0; j < loopCnt; j++ {
				sh.Println("... worker", ii, "loop", j)
				txExec(tx, "begin")
				txExec(tx, fmt.Sprintf("insert into mo11957_%d(j, k, t) values(%d, %d, 'foobarzoo')", tbn, ii, j))
				txExec(tx, fmt.Sprintf("insert into mo11957_%d(j, k, t) values(%d, %d, 'foobarzoo')", tbn, ii, j))
				txExec(tx, fmt.Sprintf("update mo11957_%d set t = 'foobarzoo updated' where j = %d and k = %d", tbn, ii, j-1))
				txExec(tx, "commit")
			}
		}(i, &wg)
	}

	wg.Wait()
	sh.Println("done")
}
