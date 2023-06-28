package main

import (
	"github.com/abiosoft/ishell/v2"

	"github.com/matrixorigin/mojo/pkg/common"
	"github.com/matrixorigin/mojo/pkg/mo"
	"github.com/matrixorigin/mojo/pkg/test"
)

func main() {
	common.ParseFlags()
	shell := ishell.New()
	shell.SetHomeHistoryPath(".mojo_history")

	// open a database connection, must be after parse flags.
	mo.Open()

	if common.Verbose {
		shell.Println("Mojo!  Seek help with mojo -help")
	}

	// Add commands.
	common.BuildCmd(shell)
	test.BuildCmd(shell)
	mo.BuildCmd(shell)

	shell.Run()
}
