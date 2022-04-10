build: configure
	cmake --build build 
configure:
	cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=${VCPKG_HOME}/scripts/buildsystems/vcpkg.cmake

.PHONY: configure clean

clean:
	rm build -r

fmt:
	clang-format -i include/*.cuh include/*.h src/*.cpp src/*.cu
