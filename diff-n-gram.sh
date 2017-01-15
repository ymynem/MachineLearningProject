mkdir -p diffngram

for n in 3 4 5 6 7 8 10 12 14
do
    echo $n
    time python3 classify.py ngram -n $n -i 2  > diffngram/$ngrami2.txt
done

