package repro

import (
	"database/sql"
	"strings"

	"github.com/abiosoft/ishell/v2"
	"github.com/matrixorigin/mojo/pkg/mo"
)

func Echo(c *ishell.Context) {
	line := strings.Join(c.RawArgs, " ~ ")
	c.Println("echo:", line)
}

func Gnuplot(c *ishell.Context) {
	mo.RunGnuplot(c, []string{"plot sin(x)"})
}

func BuildCmd(sh *ishell.Shell) {
	testCmd := &ishell.Cmd{
		Name: "!repro",
		Help: "repro a bug",
	}
	testCmd.AddCmd(&ishell.Cmd{
		Name: "moc-956",
		Help: "playing with autoincr to repor slowdown",
		Func: MoC956,
	})
	testCmd.AddCmd(&ishell.Cmd{
		Name: "mo-8871",
		Help: "mo-8871",
		Func: Mo8871,
	})
	testCmd.AddCmd(&ishell.Cmd{
		Name: "mo-11957",
		Help: "mo-11957",
		Func: Mo11957,
	})

	testCmd.AddCmd(&ishell.Cmd{
		Name: "panic",
		Help: "panicleak",
		Func: PanicLeak,
	})

	sh.AddCmd(testCmd)
}

func dbExec(db *mo.MoDB, sql string) {
	if err := db.Exec(sql); err != nil {
		panic(err)
	}
}

func txExec(tx *sql.Tx, sql string) {
	if _, err := tx.Exec(sql); err != nil {
		panic(err)
	}
}

func shtxExec(sh *ishell.Context, tx *sql.Tx, sql string) {
	rs, err := tx.Exec(sql)
	if sh != nil {
		sh.Println("exec and print rs: ", rs, "err: ", err)
	}
}

func txQueryIVal(tx *sql.Tx, sql string, params ...any) (int64, error) {
	rows, err := tx.Query(sql, params...)
	if err != nil {
		return 0, err
	}
	defer rows.Close()

	if !rows.Next() {
		return 0, nil
	}
	var ret int64
	rows.Scan(&ret)
	return ret, nil
}
