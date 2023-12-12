package repro

import (
	"flag"
	"fmt"
	"sync"

	"github.com/abiosoft/ishell/v2"
	"github.com/matrixorigin/mojo/pkg/mo"
)

func Mo11917(sh *ishell.Context) {
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

	var wg sync.WaitGroup
	wg.Add(thCnt)

	for i := 0; i < thCnt; i++ {
		go func(ii int, wg *sync.WaitGroup) {
			defer wg.Done()
			sh.Println("start worker", ii, "loop", loopCnt)
			tx, err := db.Begin()
			if err != nil {
				panic(err)
			}

			for j := 0; j < loopCnt; j++ {
				sh.Println("... worker", ii, "loop", j)
				tbn := fmt.Sprintf("xxx_%d_%d", ii, j%5)
				txExec(tx, fmt.Sprintf("drop table if exists %s", tbn))
				txExec(tx, fmt.Sprintf("create table if not exists %s(i int, j int, k int)", tbn))
				txExec(tx, fmt.Sprintf("insert into %s values (1, 2, 3)", tbn))
				txExec(tx, fmt.Sprintf("select * from %s", tbn))
			}
		}(i, &wg)
	}

	wg.Wait()
	sh.Println("done")
}
