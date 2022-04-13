build: configure
	cmake --build build 
configure:
	cmake -B build -S . -DELEMENTS_HOST_UI_LIBRARY=gtk -DELEMENTS_BUILD_EXAMPLES=OFF

#  undefined reference to symbol 'FT_Get_PS_Font_Info'
.PHONY: configure clean

clean:
	rm build -r

fmt:
	clang-format -i include/*.cuh include/*.h src/*.cpp src/*.cu
