
i=10
algorithms=('cfo')
for algorithm in "${algorithms[@]}"
do  
    for seed in 6 7 8 9 10 11 12 13 14 15
    do 
        echo "cpu $i: cfo seed$seed"
        taskset -c $i nohup python main.py \
                    --algorithm $algorithm \
                    --dataset 'vessel' \
                    --estimator 'xgboost' \
                    --metric 'rmse' \
                    --budget 14408 \
                    --seed $seed &
        i=$(($i + 1))
    done
done