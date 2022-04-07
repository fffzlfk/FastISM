clang-format
[0mTool to auto-format C/C++/Java/JavaScript/Objective-C/Protobuf/C# code.More information: https://clang.llvm.org/docs/ClangFormat.html.

 - [23;22;24;25;32mFormat a file and print the result to stdout:
[23;22;24;25;33m   clang-format {{path/to/file}}
[0m
 - [23;22;24;25;32mFormat a file in-place:
[23;22;24;25;33m   clang-format -i {{path/to/file}}
[0m
 - [23;22;24;25;32mFormat a file using a predefined coding style:
[23;22;24;25;33m   clang-format --style={{LLVM|Google|Chromium|Mozilla|WebKit}} {{path/to/file}}
[0m
 - [23;22;24;25;32mFormat a file using the .clang-format file in one of the parent directories of the source file:
[23;22;24;25;33m   clang-format --style=file {{path/to/file}}
[0m
 - [23;22;24;25;32mGenerate a custom .clang-format file:
[23;22;24;25;33m   clang-format --style={{LLVM|Google|Chromium|Mozilla|WebKit}} --dump-config > {{.clang-format}}
[0m[0m