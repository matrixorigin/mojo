ollama.wasm:
	tinygo build -o ollama.wasm -no-debug -panic=trap -scheduler=none -target wasi main.go 

clean:
	rm ollama.wasm
