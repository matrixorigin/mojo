sudoku.wasm:
	tinygo build -o sudoku.wasm -no-debug -panic=trap -scheduler=none -target wasi main.go 

gen:
	extism call sudoku.wasm sudoku_gen --wasi

solve1:
	extism call sudoku.wasm sudoku_solve --input "4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......" --wasi




