package common

import (
	"bytes"
	"database/sql"
	"encoding/csv"
	"flag"
	"fmt"
	"path/filepath"
	"regexp"
	"runtime"

	_ "github.com/go-sql-driver/mysql"
	_ "github.com/mattn/go-sqlite3"
)

var Suit string
var SuitDb string
var MoHost string
var MoUser string
var MoPasswd string
var MoDb string

func ParseArgs() {
	flag.StringVar(&Suit, "suit", "spider2", "t2sql suit")
	flag.StringVar(&SuitDb, "suitdb", "", "t2sql suit database, default to ALL")
	flag.StringVar(&MoHost, "h", "127.0.0.1:6001", "mo database host:port")
	flag.StringVar(&MoUser, "u", "dump", "mo database user")
	flag.StringVar(&MoPasswd, "p", "111", "mo database password")
	flag.StringVar(&MoDb, "d", "mysql", "initial database to conenct to")
	flag.Parse()
}

func ProjectRoot() string {
	_, file, _, _ := runtime.Caller(0)
	dir := filepath.Dir(file)
	root, _ := filepath.Abs(filepath.Join(dir, "../.."))
	return root
}

type ColInfo struct {
	Name        string
	Type        string
	OrigName    string
	OrigType    string
	Description string
}

type TableInfo struct {
	Name     string // table name
	OrigName string // original table name
	Sql      string // create table sql statement
	ColInfos []ColInfo
}

type DbInfo struct {
	Name       string // database name
	SchemaSql  string // schema.sql file path
	SqlLite    string // sqlite file path
	TableInfos []TableInfo
}

func OpenMoDB() (*sql.DB, error) {
	return sql.Open("mysql", fmt.Sprintf("%s:%s@tcp(%s)/%s", MoUser, MoPasswd, MoHost, MoDb))
}

func OpenMoDBName(dbName string) (*sql.DB, error) {
	return sql.Open("mysql", fmt.Sprintf("%s:%s@tcp(%s)/%s", MoUser, MoPasswd, MoHost, dbName))
}

func OpenSqliteDB(name string) (*sql.DB, error) {
	return sql.Open("sqlite3", name)
}

func ReadSqliteRows(file string, sqlStr string, args ...any) ([][]sql.NullString, error) {
	db, err := OpenSqliteDB(file)
	if err != nil {
		return nil, err
	}
	defer db.Close()

	rows, err := db.Query(sqlStr, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var ret [][]sql.NullString
	colNames, err := rows.Columns()
	if err != nil {
		return nil, err
	}

	ncol := len(colNames)
	datap := make([]any, ncol)
	for rows.Next() {
		data := make([]sql.NullString, ncol)
		for i := 0; i < ncol; i++ {
			datap[i] = &data[i]
		}
		err = rows.Scan(datap...)
		if err != nil {
			return nil, err
		}
		ret = append(ret, data)
	}
	return ret, nil
}

var timetzRegex = regexp.MustCompile(`^\d{4}.\d{2}.\d{2}.\d{2}:\d{2}:\d{2}`)

func ConvertSqliteToMo(value sql.NullString, tgtType string) (string, bool) {
	if !value.Valid {
		return "", false
	}

	// Nan and Inf are converted to null
	switch value.String {
	case "NaN", "inf", "\\N", "null", "NULL":
		return "", false
	}

	ret := value.String
	if (tgtType == "timestamp" || tgtType == "datetime" || tgtType == "date") && timetzRegex.MatchString(ret) {
		ret = ret[:len("2000-11-11 00:00:00")]
		bs := []byte(ret)
		bs[10] = ' '
		ret = string(bs)
	}
	return ret, true
}

func ReadSqliteCsv(file string, sqlStr string, args ...any) (string, error) {
	rows, err := ReadSqliteRows(file, sqlStr, args...)
	if err != nil {
		return "", err
	}

	buf := bytes.NewBufferString("")
	writer := csv.NewWriter(buf)

	rowStr := make([]string, len(rows[0]))
	for _, row := range rows {
		for i, col := range row {
			tmpStr, _ := ConvertSqliteToMo(col, "")
			rowStr[i] = tmpStr
		}
		writer.Write(rowStr)
	}
	writer.Flush()

	return buf.String(), nil
}

func ReadSqliteCsvBatches(file string, sqlStr string, args ...any) ([]string, error) {
	rows, err := ReadSqliteRows(file, sqlStr, args...)
	if err != nil {
		return nil, err
	}

	var ret []string

	buf := bytes.NewBufferString("")
	writer := csv.NewWriter(buf)

	rowStr := make([]string, len(rows[0]))
	for rowIdx, row := range rows {
		for i, col := range row {
			tmpStr, _ := ConvertSqliteToMo(col, "")
			rowStr[i] = tmpStr
		}
		writer.Write(rowStr)
		if (rowIdx+1)%1024 == 0 {
			writer.Flush()
			ret = append(ret, buf.String())
			buf = bytes.NewBufferString("")
			writer = csv.NewWriter(buf)
		}
	}
	writer.Flush()
	lastBatch := buf.String()
	if len(lastBatch) > 0 {
		ret = append(ret, buf.String())
	}
	return ret, nil
}

func MustExec(db *sql.DB, sql string, args ...interface{}) {
	_, err := db.Exec(sql, args...)
	if err != nil {
		panic(err)
	}
}
