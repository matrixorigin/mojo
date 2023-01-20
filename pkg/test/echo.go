package test

import (
	"os"
	"os/exec"
	"strings"

	"github.com/abiosoft/ishell/v2"
)

func Echo(c *ishell.Context) {
	line := strings.Join(c.Args, " ")
	c.Println("echo:", line)
}

func Gnuplot(c *ishell.Context) {
	f, err := os.Create("/tmp/mojo.gnuplot")
	if err != nil {
		c.Println("Error:", err)
	}

	f.WriteString("set terminal sixelgd\n")
	f.WriteString("plot sin(x)\n")
	f.Close()

	gcmd := exec.Command("/usr/bin/gnuplot", "/tmp/mojo.gnuplot")
	output, err := gcmd.Output()
	if err != nil {
		c.Println("Error:", err)
	}
	c.Println(string(output))
}

func BuildCmd(sh *ishell.Shell) {
	testCmd := &ishell.Cmd{
		Name: "test",
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
