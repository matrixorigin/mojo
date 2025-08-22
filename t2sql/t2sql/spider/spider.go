package spider

import (
	"bufio"
	"bytes"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/matrixorigin/mojo/t2sql/t2sql/common"
)

func loadOneDb(db *sql.DB, dir, dbName string) error {
	// schema file
	dbDir := filepath.Join(dir, dbName)
	schemaFile := filepath.Join(dbDir, "schema.sql")

	schema, err := os.Open(schemaFile)
	if err != nil {
		return fmt.Errorf("failed to read schema file: %s, %w", schemaFile, err)
	}
	defer schema.Close()

	buf := bytes.NewBuffer(nil)
	scanner := bufio.NewScanner(schema)
	for scanner.Scan() {
		line := scanner.Text()
		// remove all pragama stuff
		if strings.HasPrefix(line, "PRAGMA") {
			continue
		}
		buf.WriteString(line)
		buf.WriteString("\n")
	}

	common.MustExec(db, "DROP DATABASE IF EXISTS "+dbName)
	common.MustExec(db, "CREATE DATABASE "+dbName)
	common.MustExec(db, "USE "+dbName)

	if _, err := db.Exec(buf.String()); err != nil {
		return fmt.Errorf("failed to execute schema for db: %s, %w", dbName, err)
	}
	return nil
}
