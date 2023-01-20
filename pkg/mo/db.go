package mo

import (
	"database/sql"
	"fmt"
	"strings"

	_ "github.com/go-sql-driver/mysql"
	"github.com/matrixorigin/mojo/pkg/common"
	"github.com/olekukonko/tablewriter"
)

var db *sql.DB

func Open() {
	var err error
	connstr := fmt.Sprintf("%s:%s@tcp(%s:%s)/%s",
		common.MoUser, common.MoPasswd,
		common.MoHost, common.MoPort,
		common.MoDb)
	db, err = sql.Open("mysql", connstr)
	if err != nil {
		panic(err)
	}

	// go sql weird stuff
	// db.SetConnMaxIdleTime(0)
	// db.SetConnMaxLifetime(0)
	// db.SetMaxOpenConns(1)
	// db.SetMaxIdleConns(0)

	// immediately set some of the most freq used vars
	qd, err := QueryVal("select current_date")
	common.PanicIf(err)
	common.SetVar("MOJO_LOGDATE", qd)
}

func Exec(sql string, params ...any) error {
	_, err := db.Exec(sql, params...)
	return err
}

func QueryVal(sql string, params ...any) (string, error) {
	rows, err := db.Query(sql, params...)
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

func Query(sql string, params ...any) (string, error) {
	rows, err := db.Query(sql, params...)
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

func QToken(tokens []string) (string, error) {
	var tks []string
	var params []any
	for _, v := range tokens {
		if len(v) >= 2 && v[0] == ':' && v[len(v)-1] == ':' {
			vk := v[1 : len(v)-1]
			tks = append(tks, common.GetVar(vk))
		} else if len(v) >= 2 && v[0] == '?' && v[len(v)-1] == '?' {
			vk := v[1 : len(v)-1]
			tks = append(tks, "?")
			params = append(params, common.GetVar(vk))
		} else {
			tks = append(tks, v)
		}
	}
	qry := strings.Join(tks, " ")
	return Query(qry, params...)
}
