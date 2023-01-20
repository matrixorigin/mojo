package main

import (
	"github.com/abiosoft/ishell/v2"

	"github.com/matrixorigin/mojo/pkg/common"
	"github.com/matrixorigin/mojo/pkg/test"
)

func main() {
	common.ParseFlags()
	shell := ishell.New()

	if common.Verbose {
		shell.Println("Mojo!  Seek help with mojo -help")
	}

	// Add commands.
	testCmd := test.BuildCmd()
	shell.AddCmd(testCmd)

	shell.Run()
}
