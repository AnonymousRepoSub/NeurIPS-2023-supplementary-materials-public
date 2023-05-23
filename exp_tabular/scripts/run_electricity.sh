
i=0
algorithms=("cfo")
for algorithm in "${algorithms[@]}"
do  
    for seed in 6 7 8 9 10 11 12 13 14 15
    do 
        echo "cpu $i: $algorithm seed$seed"
        taskset -c $i nohup python main.py \
                    --algorithm $algorithm \
                    --dataset 'electricity' \
                    --estimator 'xgboost' \
                    --metric 'roc_auc' \
                    --budget 7208 \
                    --seed $seed &
        i=$(($i + 1))
    done
done