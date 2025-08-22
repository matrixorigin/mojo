package common

import (
	"database/sql"
)

func CheckSyntax(db *sql.DB, dbName string, sqlStr string) error {
	db.Exec("USE " + dbName)
	// check if sqlStr is valid by preparing it.   Another, probably better, way is to
	// explain it.
	stmt, err := db.Prepare(sqlStr)
	if err != nil {
		return err
	}
	defer stmt.Close()
	return nil
}

func FixSyntax(db *sql.DB, dbName string, sqlStr string, err error) (string, error) {
	// to do, right now we dont fix anything.
	return sqlStr, err
}
