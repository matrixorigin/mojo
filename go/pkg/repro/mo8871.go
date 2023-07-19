package repro

import (
	"flag"
	"fmt"
	"sync"

	"github.com/abiosoft/ishell/v2"
	"github.com/matrixorigin/mojo/pkg/mo"
)

func DelWorker(sh *ishell.Context, db *mo.MoDB, id, loop int, wg *sync.WaitGroup) {
	sh.Println("worker", id, "started")
	defer wg.Done()
	for i := 0; i < loop; i++ {
		sh.Println("del worker", id, "loop", i)
		tx, err := db.Begin()
		if err != nil {
			sh.Println("del worker", id, "loop", i, "begin failed", err)
		}

		dstr := fmt.Sprintf("update mo8871_b set k = -100 - %d where ia = %d and ib = %d and j = %d", id, id%2, id%2+1, i%5)
		_, err = tx.Exec(dstr)
		if err != nil {
			sh.Println("del worker", id, "loop", i, "insert failed", err)
		}
		err = tx.Commit()
		if err != nil {
			sh.Println("worker", id, "loop", i, "commit failed", err)
		}
		sh.Println("del worker", id, "loop", i, "done")

		if false && i%10 == 0 {
			sh.Println("worker", id, "loop", i, "flush")
			_, err = db.Query("select mo_ctl('dn', 'flush', 'repro.mo8871_b')")
			if err != nil {
				sh.Println("worker", id, "loop", i, "commit failed", err)
			}
		}
	}
}

func Worker(sh *ishell.Context, db *mo.MoDB, id, loop int, s4upd bool, wg *sync.WaitGroup) {
	sh.Println("worker", id, "started")
	defer wg.Done()
	for i := 0; i < loop; i++ {
		tx, err := db.Begin()
		if err != nil {
			sh.Println("worker", id, "loop", i, "begin failed", err)
		}

		qstr := fmt.Sprintf("select j from mo8871_a where ia = %d and ib = %d", i%5, i%5+1)
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

		_, err = tx.Exec(fmt.Sprintf("update mo8871_a set j = %d where ia = %d and ib = %d", newj, i%5, i%5+1))
		if err != nil {
			sh.Println("worker", id, "loop", i, "update failed", err)
		}

		_, err = tx.Exec(fmt.Sprintf("insert into mo8871_b values(%d, %d, %d, %d)", i%5, i%5+1, newj, id))
		if err != nil {
			sh.Println("worker", id, "loop", i, "insert failed", err)
		}

		_, err = tx.Exec(fmt.Sprintf("update mo8871_b set k = %d where ia = %d and ib = %d and j = %d", id, i%5, i%5+1, newj/2))
		err = tx.Commit()
		if err != nil {
			sh.Println("worker", id, "loop", i, "commit failed", err)
		}

		if i%100 == 0 {
			sh.Println("worker", id, "loop", i)
		}

		if false && i%10 == id%10 {
			sh.Println("worker", id, "loop", i, "flush")
			// _, err = db.Query("select mo_ctl('dn', 'merge', 'repro.mo8871_a')")
			// _, err = db.Query("select mo_ctl('dn', 'merge', 'repro.mo8871_b')")
			_, err = db.Query("select mo_ctl('dn', 'checkpoint', '')")
			if err != nil {
				sh.Println("worker", id, "loop", i, "commit failed", err)
			}
		}
	}
}

func Mo8871(sh *ishell.Context) {
	fs := flag.NewFlagSet("mo-8871", flag.ContinueOnError)
	var nth int
	var nd int
	var loop int
	var s4upd bool
	fs.IntVar(&nth, "n", 1, "number of threads")
	fs.IntVar(&nd, "d", 0, "number of delete threads")
	fs.IntVar(&loop, "loop", 10, "exec loop count")
	fs.BoolVar(&s4upd, "u", true, "if use select for update")

	if err := fs.Parse(sh.Args); err != nil {
		sh.Println()
		return
	}
	sh.Printf("MoC-956: autoincr repro, %dx%d\n", nth, loop)

	modb := mo.DefaultDB()
	modb.Exec("create database if not exists repro")

	db1, err := mo.OpenDB("16001", "repro")
	if err != nil {
		panic(err)
	}
	defer db1.Close()
	db2, err := mo.OpenDB("16002", "repro")
	if err != nil {
		panic(err)
	}
	defer db2.Close()

	db1.MustExec("drop table if exists mo8871_a")
	db1.MustExec("drop table if exists mo8871_b")
	db1.MustExec("create table mo8871_a(ia int not null, ib int not null, j int, k int, primary key(ia, ib))")
	db1.MustExec("create table mo8871_b(ia int not null, ib int not null, j int not null, k int, primary key(ia, ib, j))")
	db1.MustExec("insert into mo8871_a values(0, 1, 0, 1)")
	db1.MustExec("insert into mo8871_a values(1, 2, 0, 1)")
	db1.MustExec("insert into mo8871_a values(2, 3, 0, 1)")
	db1.MustExec("insert into mo8871_a values(3, 4, 0, 1)")
	db1.MustExec("insert into mo8871_a values(4, 5, 0, 1)")

	var wg sync.WaitGroup
	wg.Add(nth + nd)
	for i := 0; i < nth; i++ {
		sh.Println("start worker", i, "loop", loop)
		if i%2 == 0 {
			go Worker(sh, db1, i, loop, s4upd, &wg)
		} else {
			go Worker(sh, db2, i, loop, s4upd, &wg)
		}
	}

	for i := 0; i < nd; i++ {
		sh.Println("start delete worker", i, "loop", loop)
		go DelWorker(sh, db1, i, loop, &wg)
	}
	wg.Wait()

	newjs, err := db2.Query("select j from mo8871_a")
	if err != nil {
		panic(err)
	}
	sh.Println(newjs)

	insjs, err := db2.Query("select ia, ib, max(j) from mo8871_b group by ia, ib")
	if err != nil {
		panic(err)
	}
	sh.Println(insjs)
}
