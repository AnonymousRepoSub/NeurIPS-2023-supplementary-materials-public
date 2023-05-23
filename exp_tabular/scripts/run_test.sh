
i=0 
algorithms=("cfo" "hypertime")
algorithms=('bo')

for algorithm in "${algorithms[@]}"
do  
    echo "cpu $i $algorithm"
    taskset -c $i nohup python test.py \
                --algorithm $algorithm \
                --dataset 'electricity'\
                --estimator 'xgboost'\
                --metric 'roc_auc' \
                --budget 14400 \
                --shuffle \
                --size 0 &
    i=$(($i + 1))
    sleep 0.5
done
