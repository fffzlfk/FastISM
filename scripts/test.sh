for i in `seq 1 20`
do
    echo $i:
    ./build/$1 ./images/${i}.bmp
done