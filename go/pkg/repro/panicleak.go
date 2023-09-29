package repro

import (
	"database/sql"
	"flag"
	"fmt"
	"sync"

	"github.com/abiosoft/ishell/v2"
	"github.com/matrixorigin/mojo/pkg/mo"
)

func dbExec(db *mo.MoDB, sql string) {
	if err := db.Exec(sql); err != nil {
		panic(err)
	}
}

func txExec(tx *sql.Tx, sql string) {
	if _, err := tx.Exec(sql); err != nil {
		panic(err)
	}
}

func shtxExec(sh *ishell.Context, tx *sql.Tx, sql string) {
	rs, err := tx.Exec(sql)
	if sh != nil {
		sh.Println("exec and print rs: ", rs, "err: ", err)
	}
}

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

	dbExec(db, "drop table if exists panicleak")
	dbExec(db, "create table panicleak(i int not null primary key, j int, k int)")
	dbExec(db, "insert into panicleak values(0, 1, 2)")
	dbExec(db, "insert into panicleak values(1, 1, 2)")
	dbExec(db, "select enable_fault_injection()")

	// error here is ok, because the fault point may have already been created
	modb.Exec("select add_fault_point('panic', ':::', 'panic', 0, '')")

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
				txExec(tx, fmt.Sprintf("insert into panicleak values(%d, 1, 2)", ii+100))
				shtxExec(nil, tx, "select trigger_fault_point('panic')")
				txExec(tx, "commit")
			}
		}(i, &wg)
	}

	wg.Wait()
	sh.Println("done")
}
