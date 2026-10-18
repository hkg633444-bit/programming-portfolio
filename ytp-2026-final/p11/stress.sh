#!/bin/bash
# 用法: bash stress.sh [組數，預設 1000]
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -o sol sol.cpp || exit 1
g++ -O2 -std=c++17 -o brute brute.cpp || exit 1
T=${1:-1000}
for ((i=1;i<=T;i++)); do
    python3 gen.py $i > in.txt
    ./sol < in.txt > out_sol.txt
    ./brute < in.txt > out_brute.txt
    if ! diff -q out_sol.txt out_brute.txt > /dev/null; then
        echo "第 $i 組不一致 (seed=$i)，測資在 in.txt"
        echo "sol:";   cat out_sol.txt
        echo "brute:"; cat out_brute.txt
        exit 1
    fi
done
echo "通過 $T 組"
