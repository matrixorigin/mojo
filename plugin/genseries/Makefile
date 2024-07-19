genseries.wasm:
	tinygo build -o genseries.wasm -no-debug -panic=trap -scheduler=none -target wasi main.go 

run:
	extism call genseries.wasm genseries_init --input '[1, 20]' --wasi
