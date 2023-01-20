package errlog

import (
	"flag"
	"fmt"
	"strings"

	"github.com/abiosoft/ishell/v2"
	"github.com/matrixorigin/mojo/pkg/common"
	"github.com/matrixorigin/mojo/pkg/mo"
)

func logdate() string {
	qd := common.GetVar("MOJO_LOGDATE")
	qd = strings.ReplaceAll(qd, "-", "/")
	return fmt.Sprintf("__mo_filepath like '%%/%s/%%'", qd)
}

func Find(c *ishell.Context) {
	fs := flag.NewFlagSet("log.find", flag.ContinueOnError)
	var limit, offset int
	var desc bool
	var pat, ipat, rpat string

	fs.StringVar(&pat, "like", "", "like pattern")
	fs.StringVar(&ipat, "ilike", "", "ilike pattern")
	fs.StringVar(&rpat, "rlike", "", "rlike pattern")

	fs.IntVar(&limit, "n", 10, "limit of output")
	fs.IntVar(&offset, "x", 0, "offset of output")
	fs.BoolVar(&desc, "r", false, "reverse ordering")

	cols := c.Args[0]
	if err := fs.Parse(c.Args[1:]); err != nil {
		c.Println()
	}

	limitStr := fmt.Sprintf(" limit %d ", limit)
	offsetStr := ""
	if offset > 0 {
		offsetStr = fmt.Sprintf(" offset %d ", offset)
	}

	descStr := ""
	if desc {
		descStr = " desc "
	}

	qry := "select " + cols + " from system.log_info where " + logdate()
	if pat != "" {
		qry = qry + fmt.Sprintf(" and message like '%%%s%%' ", pat)
	}
	if ipat != "" {
		qry = qry + fmt.Sprintf(" and message ilike '%%%s%%' ", ipat)
	}
	if rpat != "" {
		qry = qry + fmt.Sprintf(" and message rlike '%%%s%%' ", rpat)
	}
	qry = qry + " order by timestamp " + descStr + limitStr + offsetStr

	if common.Verbose {
		c.Println("Running Query:", qry)
	}

	res, err := mo.Query(qry)
	if err != nil {
		c.Println("Error:", err)
	} else {
		c.Println(res)
	}
}

func BuildCmd(sh *ishell.Shell) {
	logCmd := &ishell.Cmd{
		Name: "log",
		Help: "log query tool",
	}
	logCmd.AddCmd(&ishell.Cmd{
		Name: "find",
		Help: "find cols conditions order by timestamp with MOJO_RESULT_LIMIT/OFFSET",
		Func: Find,
	})

	sh.AddCmd(logCmd)
}
