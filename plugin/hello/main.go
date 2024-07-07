package main

import (
    "encoding/json"
    "fmt"

    "github.com/extism/go-pdk"
)

//export mowasm_hello
func mowasm_hello() int32 {
    input := pdk.InputString()
    output := fmt.Sprintf("Hello %s!", input)
    pdk.OutputString(output)
    return 0
}

//export mowasm_add
func mowasm_add() int32 {
    input := pdk.Input()
    var args []float64
    if err := json.Unmarshal(input, &args); err != nil {
        pdk.SetError(err)
        return -1
    }

    if len(args) != 2 {
        err := fmt.Errorf("add takes two float arguments")
        pdk.SetError(err)
        return -2
    }

    output := fmt.Sprintf("%g", args[0]+args[1])
    pdk.OutputString(output)
    return 0
}

// required for WASI build
func main() {}










