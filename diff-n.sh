for n in 3 4 5 6 7 8 10 12 14
do
    echo $n
    time python3 classify.py ssk -n $n -i 2 -l 0.5 > n$ni2l0.5.txt
done

