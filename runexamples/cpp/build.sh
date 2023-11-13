# if out/ does not exist, create it
if [ ! -d "out/" ]; then
  mkdir out
fi
# copy the shared library to out/
cp ./onnx/lib/libonnxruntime.so.1.16.2 out/
cp ./tokenizer/rwkv_vocab_v20230424.txt out/

# compile the test.cpp file, linking the shared library, and setting the LD_LIBRARY_PATH to the same directory as the executable
g++ -std=c++17 -g ./test.cpp ./onnx/lib/libonnxruntime.so -o ./out/test -I./onnx/include -I./include -L./onnx/lib -l:libonnxruntime.so.1.16.2 -Wl,-rpath=./
