package mo

import (
	"database/sql"
	"fmt"
	"os"
	"strings"

	_ "github.com/go-sql-driver/mysql"
	"github.com/matrixorigin/mojo/pkg/common"
	"github.com/olekukonko/tablewriter"
)

type MoDB struct {
	db *sql.DB
}

var dfltDB *MoDB

func (db *MoDB) Close() error {
	if db.db != nil {
		err := db.db.Close()
		db.db = nil
		return err
	}
	return nil
}

func Open() error {
	var err error
	if dfltDB != nil {
		dfltDB.Close()
	}
	dfltDB, err = OpenDB(common.GetVar("MODB"))
	return err
}

func DefaultDB() *MoDB {
	return dfltDB
}

func OpenDB(dbname string) (*MoDB, error) {
	var modb MoDB
	connstr := fmt.Sprintf("%s:%s@tcp(%s:%s)/%s",
		common.GetVar("MOUSER"), common.GetVar("MOPASSWD"),
		common.GetVar("MOHOST"), common.GetVar("MOPORT"),
		dbname)

	db, err := sql.Open("mysql", connstr)
	if err != nil {
		return nil, err
	}
	// We do not want connection pooling.  Go impl is a mess.
	db.SetMaxOpenConns(1)
	modb.db = db
	return &modb, nil
}

func PyConnStr() string {
	return fmt.Sprintf("mysql+pymysql://%s:%s@%s:%s/%s",
		common.GetVar("MOUSER"), common.GetVar("MOPASSWD"),
		common.GetVar("MOHOST"), common.GetVar("MOPORT"),
		common.GetVar("MODB"))
}

func (db *MoDB) Exec(sql string, params ...any) error {
	_, err := db.db.Exec(sql, params...)
	return err
}

func (db *MoDB) Prepare(sql string) (*sql.Stmt, error) {
	return db.db.Prepare(sql)
}

func (db *MoDB) Begin() (*sql.Tx, error) {
	return db.db.Begin()
}

func (db *MoDB) QueryVal(sql string, params ...any) (string, error) {
	rows, err := db.db.Query(sql, params...)
	if err != nil {
		return "", err
	}
	defer rows.Close()

	if !rows.Next() {
		return "", nil
	}
	var ret string
	rows.Scan(&ret)
	return ret, nil
}

func (db *MoDB) Query(sql string, params ...any) (string, error) {
	rows, err := db.db.Query(sql, params...)
	if err != nil {
		return "", err
	}
	defer rows.Close()

	cols, err := rows.Columns()
	if err != nil {
		return "", err
	}

	ncol := len(cols)
	if ncol == 0 {
		return "", nil
	}

	sb := &strings.Builder{}
	tw := tablewriter.NewWriter(sb)
	tw.SetHeader(cols)
	tw.SetBorders(tablewriter.Border{Left: true, Right: true, Top: false, Bottom: false})
	tw.SetCenterSeparator("|")

	for rows.Next() {
		row := make([]interface{}, ncol)
		data := make([]string, ncol)
		for i := 0; i < ncol; i++ {
			row[i] = &data[i]
		}
		rows.Scan(row...)
		tw.Append(data)
	}

	tw.Render()
	return sb.String(), nil
}

func token2q(tokens []string) (string, []any) {
	var tks []string
	var params []any
	// poorman's macro
	for _, v := range tokens {
		if len(v) >= 2 && v[0] == ':' && v[len(v)-1] == ':' {
			// :FOO: will expand FOO
			vk := v[1 : len(v)-1]
			tks = append(tks, common.GetVar(vk))
		} else if len(v) >= 2 && v[0] == '?' && v[len(v)-1] == '?' {
			// ?FOO? will bind FOO as param
			vk := v[1 : len(v)-1]
			tks = append(tks, "?")
			params = append(params, common.GetVar(vk))
		} else if len(v) >= 2 && v[0] == '$' && v[len(v)-1] == '$' {
			// $FOO$ will become 'FOO', to work around ishell quote
			vk := v[1 : len(v)-1]
			tks = append(tks, "'"+vk+"'")
		} else {
			tks = append(tks, v)
		}
	}
	qry := strings.Join(tks, " ")
	return qry, params
}

func QToken(tokens []string) (string, error) {
	qry, params := token2q(tokens)
	return dfltDB.Query(qry, params...)
}

func QSave(tokens []string, f *os.File) error {
	sql, params := token2q(tokens)
	return qSave(dfltDB, sql, params, f)
}

func qSave(db *MoDB, sql string, params []any, f *os.File) error {
	rows, err := db.db.Query(sql, params...)
	if err != nil {
		return err
	}
	defer rows.Close()

	cols, err := rows.Columns()
	if err != nil {
		return err
	}

	ncol := len(cols)
	if ncol == 0 {
		return nil
	}

	for rows.Next() {
		row := make([]interface{}, ncol)
		data := make([]string, ncol)
		for i := 0; i < ncol; i++ {
			row[i] = &data[i]
		}
		rows.Scan(row...)
		f.WriteString(strings.Join(data, ",") + "\n")
	}
	return nil
}
