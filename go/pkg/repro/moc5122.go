package repro

import (
	"flag"

	"github.com/abiosoft/ishell/v2"
	"github.com/matrixorigin/mojo/pkg/mo"
)

func MoC5122(sh *ishell.Context) {
	fs := flag.NewFlagSet("moc-5122", flag.ContinueOnError)
	var loadMode bool
	var loadVec bool
	fs.BoolVar(&loadMode, "l", true, "load data")
	fs.BoolVar(&loadVec, "v", false, "load vector")

	var err error
	if err = fs.Parse(sh.Args); err != nil {
		sh.Println()
		return
	}
	sh.Printf("MoC-5122: load mode:%v\n", loadMode)

	if loadMode {
		sdb, err := mo.OpenDBFull(
			"freetier-01.cn-hangzhou.cluster.cn-dev.matrixone.tech",
			"snapshot_test_4eqx53:admin", "111", "6001", "test")
		if err != nil {
			panic(err)
		}

		modb := mo.DefaultDB()
		modb.Exec("create database if not exists repro")
		db, err := mo.OpenDB("6001", "repro")
		if err != nil {
			panic(err)
		}

		if err = db.Exec("drop table if exists emway"); err != nil {
			panic(err)
		}

		sql := `
		create table emway(
			md5_id varchar(255) not null primary key,
			question text,
			answer json,
			source_type varchar(255),
			content_type varchar(255),
			keyword varchar(255),
			-- question_vector vecfloat64(1024),
			allow_access varchar(511),
			allow_identities varchar(512),
			delete_flag int,
			created_at timestamp,
			updated_at timestamp
		)
		`

		if err = db.Exec(sql); err != nil {
			panic(err)
		}

		selsql := `
		select md5_id, question, answer, source_type, content_type, keyword, 
		allow_access, allow_identities, delete_flag, created_at, updated_at 
		from test.ca_comprehensive_dataset
		`

		insertsql := `
		insert into emway(md5_id, question, answer, source_type, content_type, keyword, 
		allow_access, allow_identities, delete_flag, created_at, updated_at) 
		values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		`

		sh.Printf("selsql: %s\n", selsql)
		sh.Printf("insertsql: %s\n", insertsql)

		// read from sdb and insert into db
		rows, err := sdb.Query(selsql)
		if err != nil {
			panic(err)
		}
		defer rows.Close()
		sh.Printf("selsql ok: %s\n", selsql)

		rowCnt := 0
		for rows.Next() {
			var md5_id string
			var question string
			var answer string
			var source_type string
			var content_type string
			var keyword string
			var allow_access string
			var allow_identities string
			var delete_flag int
			var created_at string
			var updated_at string

			err = rows.Scan(&md5_id, &question, &answer, &source_type, &content_type, &keyword, &allow_access, &allow_identities, &delete_flag, &created_at, &updated_at)
			if err != nil {
				sh.Printf("Error scanning row, err %v\n", err)
				panic(err)
			}
			err = db.Exec(insertsql, md5_id, question, answer, source_type, content_type, keyword, allow_access, allow_identities, delete_flag, created_at, updated_at)
			if err != nil {
				sh.Printf("Error inserting row, md5_id: %s, question: %s, answer: %s, source_type: %s, content_type: %s, keyword: %s, allow_access: %s, allow_identities: %s, delete_flag: %d, created_at: %s, updated_at: %s, err %v, err %v, err %v, err %v, err %v, err %v, err %v, err %v\n",
					md5_id, question, answer, source_type, content_type, keyword, allow_access, allow_identities, delete_flag, created_at, updated_at, err)
				panic(err)
			}
			rowCnt++
			if rowCnt%1000 == 0 {
				sh.Printf("Loaded %d rows\n", rowCnt)
			}
		}
	}
}
