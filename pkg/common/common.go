package common

import (
	"flag"
	"os"
)

var PrintHelp bool
var Verbose bool
var MoHost string
var MoPort string
var MoAccount string
var MoDb string
var MoUser string
var MoPasswd string

var MoVar map[string]string

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
	flag.BoolVar(&PrintHelp, "help", false, "print help")
	flag.BoolVar(&Verbose, "v", false, "verbose mode")
	flag.StringVar(&MoHost, "h", "", "mo host/ip, default to $MOJO_MOHOST or localhost if not set")
	flag.StringVar(&MoPort, "p", "", "mo port, default to $MOJO_MOPORT or 6001 if not set")
	flag.StringVar(&MoUser, "u", "", "mo user, default to $MOJO_MOUSER or dump if not set")
	flag.StringVar(&MoPasswd, "passwd", "", "mo password, default to $MOJO_MOPASSWD or 111 if not set")

	// parse os.Args[1:]
	flag.Parse()
	if PrintHelp {
		flag.PrintDefaults()
	}

	// fill in defaults
	MoHost = supplyDefault(MoHost, "MOJO_MOHOST", "locahost")
	MoPort = supplyDefault(MoPort, "MOJO_MOPORT", "6001")
	MoPort = supplyDefault(MoUser, "MOJO_MOUSER", "dump")
	MoPasswd = supplyDefault(MoUser, "MOJO_MOPASSWD", "111")
}
