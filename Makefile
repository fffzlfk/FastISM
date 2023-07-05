build: configure
	cmake --build build -j6

configure:
	cmake -B build -GNinja -DELEMENTS_HOST_UI_LIBRARY=gtk -DELEMENTS_BUILD_EXAMPLES=OFF

.PHONY: configure clean fmt

clean:
	rm build -r

fmt:
	clang-format -i include/*.cuh include/*.h src/*.cpp src/*.cu
