package main

import (
    "encoding/json"
    "errors"
    "strconv"

    "github.com/extism/go-pdk"
)

type state struct {
    Start int
    End int
    Step int
}

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

    var st state

    st.Start = args[0]
    st.End = args[1]
    if len(args) == 2 {
        st.Step = 1
    } else {
        st.Step = args[2]
    }

    if st.Step <= 0 {
        pdk.SetError(errors.New("genseries step must be positive"))
        return -1
    }

    save, err := json.Marshal(st) 
    if err != nil {
        pdk.SetError(err)
        return -1
    }

    pdk.SetVar("state", []byte(save))
    pdk.OutputString(string(save))
    return 0
}

//export genseries_next
func genseries_next() int32 {
    var res []string
    var st state

    stbs := pdk.GetVar("state")
    if err := json.Unmarshal(stbs, &st); err != nil {
        pdk.SetError(err)
        return -1
    }

    for i:=0; i<10; i++ {
        if st.Start < st.End {
            res = append(res, strconv.Itoa(st.Start))
            st.Start += st.Step
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

    save, err := json.Marshal(st) 
    if err != nil {
        pdk.SetError(err)
        return -1
    }

    pdk.SetVar("state", []byte(save))
    return 0
}

// required for WASI build
func main() {}
