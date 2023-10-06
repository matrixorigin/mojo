package moai

import (
	"github.com/abiosoft/ishell/v2"
)

func Moai(c *ishell.Context) {
}

// We build a mo command to send generic sql to mo db.
// As shortcut we also build select/insert/update/delete/with
// command to send sql directy.  Note that we only take
// lower case.
func BuildCmd(sh *ishell.Shell) {
	sh.AddCmd(&ishell.Cmd{
		Name: "!x",
		Help: "Start moai UI",
		Func: Moai,
	})
}
