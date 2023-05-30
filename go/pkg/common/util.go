package common

import (
	"os"
	"os/exec"

	"github.com/abiosoft/ishell/v2"
)

func PanicIf(err error) {
	if err != nil {
		panic(err)
	}
}

func RunGnuplot(c *ishell.Context, cmds []string) {
	f, err := os.Create("/tmp/mojo.gnuplot")
	if err != nil {
		c.Println("Error:", err)
	}

	f.WriteString("set timefmt \"%Y-%m-%d %H:%M:%S\"\n")
	f.WriteString("set datafile separator \",\"\n")
	f.WriteString("set terminal sixelgd\n")
	for _, cmd := range cmds {
		f.WriteString(cmd)
		f.WriteString("\n")
	}
	f.Close()

	gcmd := exec.Command("/usr/bin/gnuplot", "/tmp/mojo.gnuplot")
	output, err := gcmd.Output()
	if err != nil {
		c.Println("Error:", err)
	}
	c.Println(string(output))
}
