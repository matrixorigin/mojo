package common

import (
	"database/sql"
	"testing"
)

func TestLoadDbInfo(t *testing.T) {
	dbInfos, err := Spider2LoadDbInfo()
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Total DBs: %d\n", len(dbInfos))

	for _, dbInfo := range dbInfos {
		t.Logf("DB: %s\n", dbInfo.Name)
		for _, tableInfo := range dbInfo.TableInfos {
			t.Logf("  Table: %s\n", tableInfo.Name)
		}
	}
}

func TestSqliteRead(t *testing.T) {
	dbInfos, err := Spider2LoadDbInfo()
	f1db := dbInfos["f1"]
	if f1db == nil {
		t.Fatal("f1db not found")
	}

	t.Logf("Opening sqlite database: %s\n", f1db.SqlLite)
	sqliteDB, err := OpenSqliteDB(f1db.SqlLite)
	if err != nil {
		t.Fatal(err)
	}
	defer sqliteDB.Close()

	rows, err := sqliteDB.Query("SELECT * FROM drivers LIMIT 10")
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()

	for rows.Next() {
		data := make([]sql.NullString, 10)
		datap := make([]any, 10)
		for i := 0; i < 10; i++ {
			datap[i] = &data[i]
		}

		err = rows.Scan(datap...)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("data: %v\n", data)
	}
}

func TestCreateMoTables(t *testing.T) {
	ParseArgs()

	t.Logf("MoHost: %s\n", MoHost)
	t.Logf("MoUser: %s\n", MoUser)
	t.Logf("MoPasswd: %s\n", MoPasswd)
	t.Logf("MoDb: %s\n", MoDb)

	dbInfos, err := Spider2LoadDbInfo()
	if err != nil {
		t.Fatal(err)
	}

	f1db := dbInfos["f1"]
	if f1db == nil {
		t.Fatal("f1db not found")
	}

	mo, err := OpenMoDB()
	if err != nil {
		t.Fatal(err)
	}
	defer mo.Close()

	MustExec(mo, "DROP DATABASE IF EXISTS f1")
	MustExec(mo, "CREATE DATABASE f1")
	MustExec(mo, "USE f1")

	for _, tableInfo := range f1db.TableInfos {
		_, err = mo.Exec(tableInfo.Sql)
		if err != nil {
			t.Logf("Error creating table %s:\n%s\n", tableInfo.Name, tableInfo.Sql)
			t.Fatal(err)
		}
	}
}
