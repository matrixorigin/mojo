package main

import (
    "encoding/json"
    "errors"

    "github.com/extism/go-pdk"
)

type state struct {
    start int
    end int
    step int
}

var gState state

//export genseries_init
func genseries_init() int32 {
    input := pdk.InputString()
    var args []int
    if err := json.Unmarshal([]byte(input), &args); err != nil {
        pdk.SetError(err)
        return -1
    }

    if len(args) < 2 || len(args) > 3 {
        pdk.SetError(errors.New("genseries need 2 or 3 int arguments"))
        return -1
    }

    gState.start = args[0]
    gState.end = args[1]
    if len(args) == 2 {
        gState.step = 1
    } else {
        gState.step = args[2]
    }

    if gState.step <= 0 {
        pdk.SetError(errors.New("genseries step must be positive"))
        return -1
    }

    if gState.start <= gState.end {
        return 0
    }
    return 1
}

//export genseries_next
func genseries_next() int32 {
    var res []int
    for i:=0; i<10; i++ {
        if gState.start < gState.end {
            res = append(res, gState.start)
            gState.start += gState.step
        } else {
            break
        }
    }

    outbs, err := json.Marshal(res)
    if err != nil {
        pdk.SetError(err)
        return -1
    }

    pdk.OutputString(string(outbs))
 
    if gState.start <= gState.end {
        return 0
    }
    return 1
}

// required for WASI build
func main() {}
