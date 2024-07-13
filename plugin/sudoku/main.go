package main

import (
    "fmt"

    sudoku "github.com/eliben/go-sudoku"
    "github.com/extism/go-pdk"
)

//export sudoku_gen
func sudoku_gen() int32 {
    board := sudoku.Generate(30)
    output := sudoku.DisplayAsInput(board)
    pdk.OutputString(output)
    return 0
}

//export sudoku_solve
func sudoku_solve() int32 {
    input := pdk.Input()
    s := string(input)
    board, err := sudoku.ParseBoard(s, true)
    if err != nil {
        pdk.SetError(err)
        return -1
    }

    board, ok := sudoku.Solve(board)
    if !ok || !sudoku.IsSolved(board) {
        pdk.SetError(fmt.Errorf("cannot solve board"))
        return -2
    }

    output := sudoku.Display(board)
    pdk.OutputString(output)
    return 0
}

// required for WASI build
func main() {}










