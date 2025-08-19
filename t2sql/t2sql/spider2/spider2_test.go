package spider2

import (
	"database/sql"
	"testing"

	"github.com/matrixorigin/mojo/t2sql/t2sql/common"
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
	sqliteDB, err := common.OpenSqliteDB(f1db.SqlLite)
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
	common.ParseArgs()

	dbInfos, err := Spider2LoadDbInfo()
	if err != nil {
		t.Fatal(err)
	}

	for _, dbInfo := range dbInfos {
		t.Logf("Creating tables for %s\n", dbInfo.Name)
		err = Spider2CreateMoTables(dbInfo)
		if err != nil {
			t.Fatal(err)
		}
	}
}
