package repro

import (
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
	sh.AddCmd(testCmd)
}
