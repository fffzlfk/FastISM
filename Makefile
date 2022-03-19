build: configure
	cmake --build build

configure:
	cmake -B build

.PHONY: configure clean

clean:
	rm build -r