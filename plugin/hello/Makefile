hello.wasm:
	tinygo build -o hello.wasm -no-debug -panic=trap -scheduler=none -target wasi main.go 

hello2.wasm:
	GOOS=wasip1 GOARCH=wasm go build -tags std -o hello2.wasm .
run:
	extism call hello.wasm mowasm_hello --input world --wasi
	extism call hello.wasm mowasm_add --input "[3,5]" --wasi

err:
	extism call hello.wasm mowasm_add --input "[1,3,5]" --wasi

err2:
	extism call hello.wasm mowasm_add --input "3,5" --wasi



