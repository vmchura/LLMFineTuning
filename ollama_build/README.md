502  cd ollama_build/
503  git clone https://github.com/ggerganov/llama.cpp
505  cd llama.cpp
508  mkdir build
509  cd build
510  cmake ..
511  cmake --build . --config Release
512  cd ..
516  cd llama.cpp/
530  python3 convert_hf_to_gguf.py --outtype f16 --outfile qwen2-finetuned.gguf ../
ollama create mymodel -f Modelfile
