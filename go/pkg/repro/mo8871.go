package repro

import (
	"flag"
	"fmt"
	"sync"

	"github.com/abiosoft/ishell/v2"
	"github.com/matrixorigin/mojo/pkg/mo"
)

func Worker(sh *ishell.Context, db *mo.MoDB, id, loop int, s4upd bool, wg *sync.WaitGroup) {
	sh.Println("worker", id, "started")
	defer wg.Done()
	for i := 0; i < loop; i++ {
		tx, err := db.Begin()
		if err != nil {
			sh.Println("worker", id, "loop", i, "begin failed", err)
		}

		qstr := fmt.Sprintf("select j from mo8871_a where i = %d", i%5)
		if s4upd {
			qstr += " for update"
		}

		rows, err := tx.Query(qstr)
		if err != nil {
			sh.Println("worker", id, "loop", i, "select for update", err)
		}
		if !rows.Next() {
			sh.Println("worker", id, "loop", i, "select for update not found", err)
		}
		var oldj int
		rows.Scan(&oldj)
		newj := oldj + 1
		rows.Close()

		sh.Println("worker", id, "loop", i, "oldj", oldj, "newj", newj)

		_, err = tx.Exec(fmt.Sprintf("update mo8871_a set j = %d where i = %d", newj, i%5))
		if err != nil {
			sh.Println("worker", id, "loop", i, "update failed", err)
		}

		_, err = tx.Exec(fmt.Sprintf("insert into mo8871_b values(%d, %d, %d)", i%5, newj, id))
		if err != nil {
			sh.Println("worker", id, "loop", i, "insert failed", err)
		}
		err = tx.Commit()
		if err != nil {
			sh.Println("worker", id, "loop", i, "commit failed", err)
		}
	}
}

func Mo8871(sh *ishell.Context) {
	fs := flag.NewFlagSet("mo-8871", flag.ContinueOnError)
	var nth int
	var loop int
	var s4upd bool
	fs.IntVar(&nth, "n", 1, "number of threads")
	fs.IntVar(&loop, "loop", 10, "exec loop count")
	fs.BoolVar(&s4upd, "u", true, "if use select for update")

	if err := fs.Parse(sh.Args); err != nil {
		sh.Println()
		return
	}
	sh.Printf("MoC-956: autoincr repro, %dx%d\n", nth, loop)

	modb := mo.DefaultDB()
	modb.Exec("create database if not exists repro")

	db, err := mo.OpenDB("repro")
	if err != nil {
		panic(err)
	}
	defer db.Close()

	db.MustExec("drop table if exists mo8871_a")
	db.MustExec("drop table if exists mo8871_b")
	db.MustExec("create table mo8871_a(i int not null primary key, j int, k int)")
	db.MustExec("create table mo8871_b(i int not null, j int not null, k int, primary key(i, j))")
	db.MustExec("insert into mo8871_a values(0, 0, 1)")
	db.MustExec("insert into mo8871_a values(1, 0, 1)")
	db.MustExec("insert into mo8871_a values(2, 0, 1)")
	db.MustExec("insert into mo8871_a values(3, 0, 1)")
	db.MustExec("insert into mo8871_a values(4, 0, 1)")

	var wg sync.WaitGroup
	wg.Add(nth)
	for i := 0; i < nth; i++ {
		sh.Println("start worker", i, "loop", loop)
		go Worker(sh, db, i, loop, s4upd, &wg)
	}
	wg.Wait()

	newjs, err := db.Query("select j from mo8871_a")
	if err != nil {
		panic(err)
	}
	sh.Println(newjs)

	insjs, err := db.Query("select max(j) from mo8871_b group by i")
	if err != nil {
		panic(err)
	}
	sh.Println(insjs)
}
