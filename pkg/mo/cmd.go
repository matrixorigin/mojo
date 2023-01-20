package mo

import (
	"github.com/abiosoft/ishell/v2"
)

func MoCmd(c *ishell.Context) {
	res, err := QToken(c.Args)
	if err != nil {
		c.Println("Error:", err)
	} else {
		c.Println(res)
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
		Name: "mo",
		Help: "genric mo command/sql",
		Func: MoCmd,
	})
	sh.AddCmd(&ishell.Cmd{
		Name: "select",
		Help: "select query",
		Func: MoSelect,
	})
	sh.AddCmd(&ishell.Cmd{
		Name: "with",
		Help: "with select query",
		Func: MoWith,
	})
	sh.AddCmd(&ishell.Cmd{
		Name: "insert",
		Help: "insert query",
		Func: MoInsert,
	})
	sh.AddCmd(&ishell.Cmd{
		Name: "update",
		Help: "update query",
		Func: MoUpdate,
	})
	sh.AddCmd(&ishell.Cmd{
		Name: "delete",
		Help: "delete query",
		Func: MoDelete,
	})

}
