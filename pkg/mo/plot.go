package mo

import (
	"flag"
	"fmt"
	"strings"

	"github.com/abiosoft/ishell/v2"
	"github.com/matrixorigin/mojo/pkg/common"
)

func XyPlot(c *ishell.Context) {
	fs := flag.NewFlagSet("mo.plot", flag.ContinueOnError)
	var ifn string
	var xt, eval bool
	fs.StringVar(&ifn, "i", "/tmp/mojo.data", "input file")
	fs.BoolVar(&xt, "xt", false, "x series is time")
	fs.BoolVar(&eval, "e", false, "eval gnuplot command")

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
		if !eval {
			cmds = append(cmds, fmt.Sprintf("plot \"%s\" %s", ifn, strings.Join(fs.Args(), " ")))
		} else {
			for _, cmd := range fs.Args() {
				if strings.HasPrefix(cmd, "using") {
					cmds = append(cmds, fmt.Sprintf("plot \"%s\" %s", ifn, cmd))
				} else {
					cmds = append(cmds, cmd)
				}
			}
		}
	}

	common.RunGnuplot(c, cmds)
}
