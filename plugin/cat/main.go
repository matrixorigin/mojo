package main

import (
	pdk "github.com/extism/go-pdk"
)

//export cat
func cat() int32 {
	input := pdk.InputString()
	header, ok := pdk.GetConfig("header")
	var result string
	if ok {
		result += header
	}

	result += input

	footer, ok := pdk.GetConfig("footer")
	if ok {
		result += footer
	}
	pdk.OutputString(result)
	return 0
}

func main() {}
