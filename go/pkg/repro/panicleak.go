package repro

import (
	"flag"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/abiosoft/ishell/v2"
	"github.com/matrixorigin/mojo/pkg/mo"
)

func PanicLeak(sh *ishell.Context) {
	fs := flag.NewFlagSet("panicleak", flag.ContinueOnError)
	var thCnt int
	var loopCnt int
	var createTbl bool

	fs.IntVar(&thCnt, "t", 20, "number of threads")
	fs.IntVar(&loopCnt, "n", 10000, "loop count")
	fs.BoolVar(&createTbl, "c", false, "create table")

	var err error

	if err = fs.Parse(sh.Args); err != nil {
		sh.Println()
		return
	}
	sh.Printf("PanicLeak: thread count:%d, loop:%d\n", thCnt, loopCnt)

	modb := mo.DefaultDB()
	modb.Exec("create database if not exists repro")
	db, err := mo.OpenDB("6001", "repro")
	if err != nil {
		panic(err)
	}

	if createTbl {
		sh.Printf("PanicLeak: drop table\n")
		dbExec(db, "drop table if exists panicleak")
		sh.Printf("PanicLeak: create table\n")
		dbExec(db, "create table panicleak(i int not null, j int not null, name varchar(10) not null, k int, primary key (i, j))")
		sh.Printf("PanicLeak: insert data\n")
		dbExec(db, "insert into panicleak values(0, 0, 'foo', 1)")
		dbExec(db, "insert into panicleak values(0, 1, 'foo', 1)")
		dbExec(db, "insert into panicleak values(0, 2, 'foo', 1)")
		dbExec(db, "insert into panicleak values(0, 3, 'foo', 1)")
		dbExec(db, "insert into panicleak values(0, 4, 'foo', 1)")
		dbExec(db, "insert into panicleak values(0, 5, 'foo', 1)")
		dbExec(db, "insert into panicleak values(0, 6, 'foo', 1)")
		dbExec(db, "insert into panicleak values(0, 7, 'foo', 1)")
		dbExec(db, "insert into panicleak values(0, 8, 'foo', 1)")
		dbExec(db, "insert into panicleak values(0, 9, 'foo', 1)")
		dbExec(db, "insert into panicleak values(1, 0, 'foo', 1)")
		dbExec(db, "insert into panicleak values(1, 1, 'foo', 1)")
		dbExec(db, "insert into panicleak values(1, 2, 'foo', 1)")
		dbExec(db, "insert into panicleak values(1, 3, 'foo', 1)")
		dbExec(db, "insert into panicleak values(1, 4, 'foo', 1)")
		dbExec(db, "insert into panicleak values(1, 5, 'foo', 1)")
		dbExec(db, "insert into panicleak values(1, 6, 'foo', 1)")
		dbExec(db, "insert into panicleak values(1, 7, 'foo', 1)")
		dbExec(db, "insert into panicleak values(1, 8, 'foo', 1)")
		dbExec(db, "insert into panicleak values(1, 9, 'foo', 1)")
	}

	sh.Printf("PanicLeak: start background flush\n")
	go func() {
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		for {
			rn := r.Intn(1000)
			time.Sleep(time.Duration(rn) * time.Millisecond)
			if rn%5 == 0 {
				sh.Println("checkpoin ... ...\n")
				dbExec(db, "select mo_ctl('dn', 'checkpoint', '')")
			} else {
				sh.Print("flush ... ...\n")
				dbExec(db, "select mo_ctl('dn', 'flush', 'repro.panicleak')")
			}
		}
	}()

	var wg sync.WaitGroup
	wg.Add(thCnt)

	for i := 0; i < thCnt; i++ {
		go func(ii int, wg *sync.WaitGroup) {
			defer wg.Done()
			sh.Println("start worker", ii)
			tx, err := db.Begin()
			if err != nil {
				panic(err)
			}

			r := rand.New(rand.NewSource(int64(ii)))

			for j := 0; j < loopCnt; j++ {
				sh.Println("... worker", ii, "loop", j)
				txExec(tx, "begin")

				pki := r.Intn(2)
				// pkj := r.Intn(10)

				_, err := txQueryIVal(tx, "select k from panicleak where i = 0 and j = ? for update", pki)
				if err != nil {
					panic(err)
				}

				sleepMs := time.Duration(r.Intn(1000))
				time.Sleep(sleepMs * time.Microsecond)

				txExec(tx, fmt.Sprintf("update panicleak set k = k + 1 where i = 0 and j = %d", pki))
				txExec(tx, "commit")
			}
		}(i, &wg)
	}

	wg.Wait()
	sh.Println("done")
}
