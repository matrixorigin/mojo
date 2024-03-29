package repro

import (
	"database/sql"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/abiosoft/ishell/v2"
	"github.com/matrixorigin/mojo/pkg/mo"
)

func MoC956(sh *ishell.Context) {
	fs := flag.NewFlagSet("moc-956", flag.ContinueOnError)
	var batchMode string
	var txnMode string
	var singleTbl bool
	var dumpFn string
	fs.StringVar(&batchMode, "b", "prepare", "batch, single, prepare")
	fs.StringVar(&txnMode, "t", "begin", "auto, begin, badbegin")
	fs.BoolVar(&singleTbl, "s", false, "single table")
	fs.StringVar(&dumpFn, "d", "", "dump sql commands")

	var err error

	if err = fs.Parse(sh.Args); err != nil {
		sh.Println()
		return
	}
	sh.Printf("MoC-956: autoincr repro, batchmode:%s, txnmode:%v\n", batchMode, txnMode)

	var dumpF *os.File
	if dumpFn != "" {
		// open dumpFile to write
		dumpF, err = os.Create(dumpFn)
		if err != nil {
			panic(err)
		}
		defer dumpF.Close()

		dumpF.WriteString("use repro;\n")
	}

	modb := mo.DefaultDB()
	modb.Exec("create database if not exists repro")

	db, err := mo.OpenDB("6001", "repro")
	if err != nil {
		panic(err)
	}

	for i := 0; i < 10; i++ {
		sh.Println("drop table table:", i)
		dropTbl := fmt.Sprintf("drop table if exists tbl_%d", i)
		if err := db.Exec(dropTbl); err != nil {
			panic(err)
		}
	}

	tot := 10
	if singleTbl {
		tot = 1
	}

	for i := 0; i < tot; i++ {
		sh.Println("create table table:", i)

		start := time.Now()
		crTbl := fmt.Sprintf("create table tbl_%d(i int, j int, k int)", i)
		if dumpF != nil {
			dumpF.WriteString(crTbl + ";\n")
		}
		err = db.Exec(crTbl)
		if err != nil {
			panic(err)
		}

		sh.Println("insert into table:", i)
		if batchMode == "batch" {
			ins := fmt.Sprintf("insert into tbl_%d select result, result +1, result+2 from generate_series(1, 10000) tmpt", i)
			if dumpF != nil {
				dumpF.WriteString(ins + "\n")
			}
			err = db.Exec(ins)
			if err != nil {
				panic(err)
			}
		} else if batchMode == "single" {
			// so slow, that I will 1000 instead of 10000
			var tx *sql.Tx
			if txnMode == "begin" {
				tx, err = db.Begin()
				if err != nil {
					panic(err)
				}
			} else if txnMode == "badbegin" {
				db.Exec("begin")
			}

			for j := 0; j < 1000; j++ {
				ins := fmt.Sprintf("insert into tbl_%d values(%d, %d, %d)", i, j, j+1, j+2)
				if dumpF != nil {
					dumpF.WriteString(ins + ";\n")
				}
				if tx == nil {
					err = db.Exec(ins)
				} else {
					_, err = tx.Exec(ins)
				}
				if err != nil {
					panic(err)
				}
			}

			if txnMode == "begin" {
				tx.Commit()
			} else if txnMode == "badbegin" {
				db.Exec("commit")
			}
		} else if batchMode == "prepare" {
			// so slow, that I will 1000 instead of 10000
			var stmt *sql.Stmt
			var tx *sql.Tx
			var err error

			if txnMode == "begin" {
				tx, err = db.Begin()
				if err != nil {
					panic(err)
				}
				stmt, err = tx.Prepare(fmt.Sprintf("insert into tbl_%d values(?, ?, ?)", i))
			} else {
				stmt, err = db.Prepare(fmt.Sprintf("insert into tbl_%d values(?, ?, ?)", i))
				if err != nil {
					panic(err)
				}
				defer stmt.Close()
				if txnMode == "badbegin" {
					db.Exec("begin")
				}
			}

			for j := 0; j < 1000; j++ {
				_, err = stmt.Exec(j, j+1, j+2)
				if err != nil {
					panic(err)
				}
			}

			if txnMode == "begin" {
				stmt.Close()
				tx.Commit()
			} else if txnMode == "badbegin" {
				db.Exec("commit")
				stmt.Close()
			}
		} else {
			sh.Println("unknown batch mode", batchMode)
		}

		duration := time.Since(start)
		sh.Printf("Iter %d: Time %v\n", i, duration)

		// check
		cnt, err := db.Query(fmt.Sprintf("select count(*) from tbl_%d", i))
		sh.Println("check table:", i, "count", cnt, err)
	}
}
