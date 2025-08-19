package common

import (
	"database/sql"
	"flag"
	"fmt"
	"path/filepath"
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
	Description string
}

type TableInfo struct {
	Name     string // table name
	Sql      string // create table sql statement
	ColInfos []ColInfo
}

type DbInfo struct {
	Name       string // database name
	SchemaSql  string // schema.sql file path
	SqlLite    string // sqlite file path
	TableInfos []TableInfo
}

func LoadDbInfo() (map[string]*DbInfo, error) {
	switch Suit {
	case "spider2":
		return Spider2LoadDbInfo()
	default:
		return nil, fmt.Errorf("unknown suit: %s", Suit)
	}
}

func OpenMoDB() (*sql.DB, error) {
	return sql.Open("mysql", fmt.Sprintf("%s:%s@tcp(%s)/%s", MoUser, MoPasswd, MoHost, MoDb))
}

func OpenSqliteDB(name string) (*sql.DB, error) {
	return sql.Open("sqlite3", name)
}

func MustExec(db *sql.DB, sql string, args ...interface{}) {
	_, err := db.Exec(sql, args...)
	if err != nil {
		panic(err)
	}
}
