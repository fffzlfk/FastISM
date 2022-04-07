build: configure
	cmake --build build

configure:
	cmake -B build

.PHONY: configure clean

clean:
	rm build -r

fmt:
	clang-format -i include/*.cuh include/*.h src/*.cpp src/*.cu
