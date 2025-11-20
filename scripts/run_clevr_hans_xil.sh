

for model in "Qwen2.5-VL-7B-Instruct" "InternVL3-8B" "InternVL3-14B" "Qwen3-VL-30B-A3B-Instruct"
do
    for i in 0 1 2
    do

        for distribution in "naive_weighted"
        do

            CUDA_VISIBLE_DEVICES=$1 python xil_experiment.py --dataset "CLEVR-Hans3-confounded" --model $model --search_timeout 10 \
            --n_objects 10 --n_properties 10 --n_actions 0 --max_program_depth 6 \
            --seed $i --variable_distribution $distribution --max_imgs 10

            CUDA_VISIBLE_DEVICES=$1 python xil_experiment.py --dataset "CLEVR-Hans3-confounded" --model $model --search_timeout 10 \
            --n_objects 10 --n_properties 10 --n_actions 0  --max_program_depth 6 \
            --seed $i --variable_distribution $distribution --max_imgs 10 --xil_remove_confounders

            CUDA_VISIBLE_DEVICES=$1 python xil_experiment.py --dataset "CLEVR-Hans3-confounded" --model $model --search_timeout 10 \
            --n_objects 10 --n_properties 10 --n_actions 0 --max_program_depth 6 \
            --seed $i --variable_distribution $distribution --max_imgs 10 --xil_add_functions

            CUDA_VISIBLE_DEVICES=$1 python xil_experiment.py --dataset "CLEVR-Hans3-confounded" --model $model --search_timeout 10 \
            --n_objects 10 --n_properties 10 --n_actions 0 --max_program_depth 6 \
            --seed $i --variable_distribution $distribution --max_imgs 10 --xil_remove_confounders --xil_add_functions

        done
    done
done