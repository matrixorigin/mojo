package common

import (
	"flag"
	"os"
	"strings"

	"github.com/abiosoft/ishell/v2"
)

var PrintHelp bool
var Verbose bool

var moVar map[string]string

func supplyDefault(v, name, dflt string) string {
	if v != "" {
		return v
	}
	if ev := os.Getenv(name); ev != "" {
		return ev
	}
	return dflt
}

func ParseFlags() {
	var moHost, moPort, moUser, moPasswd, moDb string
	flag.BoolVar(&PrintHelp, "help", false, "print help")
	flag.BoolVar(&Verbose, "v", false, "verbose mode")
	flag.StringVar(&moHost, "h", "", "mo host/ip, default to $MOJO_MOHOST or localhost if not set")
	flag.StringVar(&moPort, "p", "", "mo port, default to $MOJO_MOPORT or 6001 if not set")
	flag.StringVar(&moUser, "u", "", "mo user, default to $MOJO_MOUSER or dump if not set")
	flag.StringVar(&moPasswd, "passwd", "", "mo password, default to $MOJO_MOPASSWD or 111 if not set")
	flag.StringVar(&moDb, "d", "", "mo database, default to $MOJO_MODB or mysql if not set")

	// parse os.Args[1:]
	flag.Parse()
	if PrintHelp {
		flag.PrintDefaults()
	}

	moVar = make(map[string]string)
	// fill in defaults
	moVar["MOHOST"] = supplyDefault(moHost, "MOJO_MOHOST", "localhost")
	moVar["MOPORT"] = supplyDefault(moPort, "MOJO_MOPORT", "6001")
	moVar["MOUSER"] = supplyDefault(moUser, "MOJO_MOUSER", "dump")
	moVar["MOPASSWD"] = supplyDefault(moPasswd, "MOJO_MOPASSWD", "111")
	moVar["MODB"] = supplyDefault(moDb, "MOJO_MODB", "mysql")

}

// command: set XXX value
func SetVar(k, v string) {
	if v == "" {
		delete(moVar, k)
	} else {
		moVar[k] = v
	}
}
func SetCmd(c *ishell.Context) {
	if len(c.Args) == 0 {
		return
	}
	k := c.Args[0]
	v := strings.Join(c.Args[1:], " ")
	SetVar(k, v)
}

func GetVar(k string) string {
	return moVar[k]
}

func GetVarOr(k, dflt string) string {
	if moVar[k] == "" {
		return dflt
	}
	return moVar[k]
}

func ShowCmd(c *ishell.Context) {
	if len(c.Args) == 0 {
		for k, v := range moVar {
			c.Println(k, " = ", v)
		}
		return
	}
	k := c.Args[0]
	c.Println(k, " = ", GetVar(k))
}

func BuildCmd(sh *ishell.Shell) {
	sh.AddCmd(&ishell.Cmd{
		Name: "set",
		Help: "set variable",
		Func: SetCmd,
	})
	sh.AddCmd(&ishell.Cmd{
		Name: "show",
		Help: "show variable value",
		Func: ShowCmd,
	})
}
