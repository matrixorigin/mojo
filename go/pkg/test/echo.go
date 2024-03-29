package test

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
		Name: "!test",
		Help: "dev/test only",
	}
	testCmd.AddCmd(&ishell.Cmd{
		Name: "echo",
		Help: "echo ping pong",
		Func: Echo,
	})
	testCmd.AddCmd(&ishell.Cmd{
		Name: "plot",
		Help: "gnuplot sixel output",
		Func: Gnuplot,
	})
	sh.AddCmd(testCmd)
}
