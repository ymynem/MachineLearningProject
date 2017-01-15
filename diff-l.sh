mkdir -p diffl

for l in 0.01 0.03 0.05 0.07 0.09 0.1 0.3 0.5 0.7 0.9 
do
    echo $l
    time python3 classify.py ssk -n 5 -i 2 -l $l > diffl/n5i2lv${l}.txt
done

