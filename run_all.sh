max_ep_len="10 20 30 40 50 75 100 150 200"
norm_reward="1000000 100000 10000 1000 100"

for max in $max_ep_len
do
    for norm in $norm_reward
        do
            mkdir -p out/
            run_cmd="run_rl.sh ${max} ${norm}"
            sbatch_cmd="sbatch ${run_cmd}"
            cmd="$sbatch_cmd"
            echo -e "${cmd}"
            ${cmd}
            sleep 1
        done
done