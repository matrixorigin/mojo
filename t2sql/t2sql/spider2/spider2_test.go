package spider2

import (
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

func TestSqliteReadF1(t *testing.T) {
	dbInfos, err := Spider2LoadDbInfo()
	if err != nil {
		t.Fatal(err)
	}

	f1db := dbInfos["f1"]
	if f1db == nil {
		t.Fatal("f1db not found")
	}

	t.Logf("Opening sqlite database: %s\n", f1db.SqlLite)
	rows, err := common.ReadSqliteRows(f1db.SqlLite, "SELECT * FROM drivers LIMIT 5")
	if err != nil {
		t.Fatal(err)
	}

	for _, row := range rows {
		t.Logf("row: %v\n", row)
	}

	csv, err := common.ReadSqliteCsv(f1db.SqlLite, "SELECT * FROM drivers LIMIT 5")
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("csv: %s\n", csv)
}

func TestCreateMoDB(t *testing.T) {
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

func TestLoadMoDB(t *testing.T) {
	common.ParseArgs()

	dbInfos, err := Spider2LoadDbInfo()
	if err != nil {
		t.Fatal(err)
	}

	for _, dbInfo := range dbInfos {
		t.Logf("Loading %s\n", dbInfo.Name)
		err = LoadMoDB(dbInfo)
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestLoadQueries(t *testing.T) {
	common.ParseArgs()

	err := LoadQueries()
	if err != nil {
		t.Fatal(err)
	}
}

func TestLoadMoGold(t *testing.T) {
	common.ParseArgs()

	err := LoadMoGold()
	if err != nil {
		t.Fatal(err)
	}
}
