build: configure
	cmake --build build

configure:
	cmake -B build -GNinja

.PHONY: configure clean

clean:
	rm build -r
