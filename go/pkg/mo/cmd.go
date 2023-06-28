package mo

import (
	"flag"
	"os"

	"github.com/abiosoft/ishell/v2"
)

func MoConnect(c *ishell.Context) {
	if err := Open(); err != nil {
		c.Println("Error:", err)
	}
}

func MoCmd(c *ishell.Context) {
	res, err := QToken(c.Args)
	if err != nil {
		c.Println("Error:", err)
	} else {
		c.Println(res)
	}
}

func MoSave(c *ishell.Context) {
	fs := flag.NewFlagSet("mo.save", flag.ContinueOnError)
	var ofn string
	fs.StringVar(&ofn, "o", "/tmp/mojo.data", "save to output file")

	if err := fs.Parse(c.Args); err != nil {
		c.Println()
	}

	f, err := os.Create(ofn)
	if err != nil {
		c.Println("Error:", err)
	}
	defer f.Close()

	if err = QSave(fs.Args(), f); err != nil {
		c.Println("Error:", err)
	}
}

func MoSelect(c *ishell.Context) {
	tks := append([]string{"select"}, c.Args...)
	res, err := QToken(tks)
	if err != nil {
		c.Println("Error:", err)
	} else {
		c.Println(res)
	}
}

func MoWith(c *ishell.Context) {
	tks := append([]string{"with"}, c.Args...)
	res, err := QToken(tks)
	if err != nil {
		c.Println("Error:", err)
	} else {
		c.Println(res)
	}
}

func MoInsert(c *ishell.Context) {
	tks := append([]string{"insert"}, c.Args...)
	res, err := QToken(tks)
	if err != nil {
		c.Println("Error:", err)
	} else {
		c.Println(res)
	}
}

func MoDelete(c *ishell.Context) {
	tks := append([]string{"delete"}, c.Args...)
	res, err := QToken(tks)
	if err != nil {
		c.Println("Error:", err)
	} else {
		c.Println(res)
	}
}

func MoUpdate(c *ishell.Context) {
	tks := append([]string{"update"}, c.Args...)
	res, err := QToken(tks)
	if err != nil {
		c.Println("Error:", err)
	} else {
		c.Println(res)
	}
}

// We build a mo command to send generic sql to mo db.
// As shortcut we also build select/insert/update/delete/with
// command to send sql directy.  Note that we only take
// lower case.
func BuildCmd(sh *ishell.Shell) {
	sh.AddCmd(&ishell.Cmd{
		Name: "!connect",
		Help: "connect to database",
		Func: MoConnect,
	})
	sh.AddCmd(&ishell.Cmd{
		Name: "!mo",
		Help: "genric mo command/sql",
		Func: MoCmd,
	})
	sh.AddCmd(&ishell.Cmd{
		Name: "!select",
		Help: "select query",
		Func: MoSelect,
	})
	sh.AddCmd(&ishell.Cmd{
		Name: "!with",
		Help: "with select query",
		Func: MoWith,
	})
	sh.AddCmd(&ishell.Cmd{
		Name: "!insert",
		Help: "insert query",
		Func: MoInsert,
	})
	sh.AddCmd(&ishell.Cmd{
		Name: "!update",
		Help: "update query",
		Func: MoUpdate,
	})
	sh.AddCmd(&ishell.Cmd{
		Name: "!delete",
		Help: "delete query",
		Func: MoDelete,
	})
	sh.AddCmd(&ishell.Cmd{
		Name: "!save",
		Help: "save query result",
		Func: MoSave,
	})
	sh.AddCmd(&ishell.Cmd{
		Name: "!plot",
		Help: "plot saved query result",
		Func: XyPlot,
	})
}
