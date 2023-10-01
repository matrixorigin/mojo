package repro

import (
	"flag"
	"fmt"
	"sync"

	"github.com/abiosoft/ishell/v2"
	"github.com/matrixorigin/mojo/pkg/mo"
)

func PanicLeak(sh *ishell.Context) {
	fs := flag.NewFlagSet("panicleak", flag.ContinueOnError)
	var thCnt int
	var loopCnt int
	fs.IntVar(&thCnt, "t", 10, "number of threads")
	fs.IntVar(&loopCnt, "n", 200, "loop count")

	var err error

	if err = fs.Parse(sh.Args); err != nil {
		sh.Println()
		return
	}
	sh.Printf("PanicLeak: thread count:%d, loop:%n\n", thCnt, loopCnt)

	modb := mo.DefaultDB()
	modb.Exec("create database if not exists repro")
	db, err := mo.OpenDB("6001", "repro")
	if err != nil {
		panic(err)
	}

	// dbExec(db, "drop table if exists panicleak")
	// dbExec(db, "create table panicleak(i int not null, name varchar(10) not null, j int, k int, primary key (i, name))")
	// dbExec(db, "insert into panicleak values(0, 'foo', 1, 2)")
	// dbExec(db, "insert into panicleak values(1, 'bar', 1, 2)")

	//
	// dbExec(db, "select enable_fault_injection()")
	// error here is ok, because the fault point may have already been created
	// modb.Exec("select add_fault_point('panic', ':::', 'panic', 0, '')")
	//

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
				txExec(tx, "begin")
				if true {
					txExec(tx, fmt.Sprintf("insert into panicleak values (%d, 'foobarzoo', 0, 0)", ii*10000+j))
				} else if (ii+j)%2 == 0 {
					jval, err := txQueryIVal(tx, "select j from panicleak where i = 0 and name = 'foo' for update")
					if err != nil {
						panic(err)
					}
					txExec(tx, fmt.Sprintf("update panicleak set j = %d where i = 0 and name = 'foo'", jval+1))
				} else {
					jval, err := txQueryIVal(tx, "select j from panicleak where i = 1 and name = 'bar' for update")
					if err != nil {
						panic(err)
					}
					txExec(tx, fmt.Sprintf("update panicleak set j = %d where i = 1 and name = 'bar'", jval+1))
				}

				txExec(tx, "commit")

				if (j+ii)%10 == 0 {
					sh.Println("worker flush", ii, "loop", j)
					txExec(tx, "select mo_ctl('dn', 'flush', 'repro.panicleak')")
				}
			}
		}(i, &wg)
	}

	wg.Wait()
	sh.Println("done")
}
