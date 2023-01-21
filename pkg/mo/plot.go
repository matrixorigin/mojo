package mo

import (
	"flag"
	"fmt"

	"github.com/abiosoft/ishell/v2"
	"github.com/matrixorigin/mojo/pkg/common"
)

func XyPlot(c *ishell.Context) {
	fs := flag.NewFlagSet("mo.plot", flag.ContinueOnError)
	var ifn string
	var xt bool
	fs.StringVar(&ifn, "i", "/tmp/mojo.data", "input file")
	fs.BoolVar(&xt, "xt", false, "x series is time")

	if err := fs.Parse(c.Args); err != nil {
		c.Println()
	}

	var cmds []string
	if xt {
		cmds = append(cmds, "set xdata time")
	}

	if fs.NArg() == 0 {
		cmds = append(cmds, fmt.Sprintf("plot \"%s\" using 1:2", ifn))
	} else {
		cmds = append(cmds, fs.Args()...)
	}

	common.RunGnuplot(c, cmds)
}
