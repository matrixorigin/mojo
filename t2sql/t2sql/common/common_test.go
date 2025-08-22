package common

import (
	"strings"
	"testing"
)

func TestProjectRoot(t *testing.T) {
	root := ProjectRoot()
	if !strings.HasSuffix(root, "mojo/t2sql") {
		t.Errorf("ProjectRoot() = %s, want %s", root, "mojo/t2sql")
	}
}

func TestDbExec(t *testing.T) {
	ParseArgs()

	db, err := OpenMoDB()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	MustExec(db, "DROP DATABASE IF EXISTS test")
	MustExec(db, "CREATE DATABASE test")
	MustExec(db, "USE test")
	MustExec(db, `CREATE TABLE test (id INT, name VARCHAR(255)); 
					INSERT INTO test (id, name) VALUES (1, 'test'); 
					INSERT INTO test (id, name) VALUES (2, 'test2'); 
					INSERT INTO test (id, name) VALUES (3, 'test3'); 
					`)

	rows, err := db.Query("SELECT * FROM test")
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()

	rowCnt := 0
	for rows.Next() {
		var id int
		var name string
		err = rows.Scan(&id, &name)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("id: %d, name: %s\n", id, name)
		rowCnt++
	}

	if rowCnt != 3 {
		t.Fatal("rowCnt should be 3")
	}
}
