

for i in 0
do
    # for model in "InternVL3-8B" "InternVL3-14B" "Qwen2.5-VL-7B-Instruct" "Qwen3-VL-30B-A3B-Instruct" 
    for model in "Kimi-VL-A3B-Instruct"
    # for model in "Ovis2.5-9B"
    do
    for distribution in "naive_weighted" "positive_ratio"
        do
        for max_program_depth in 5
            do

                CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "bongard-hoi-max-img" --model $model --search_timeout 10 \
                --n_objects 10 --n_properties 5 --n_actions 10 --max_program_depth 4 \
                --seed $i --variable_distribution $distribution --max_imgs 10 --no_sampling

                CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "CLEVR-Hans3-unconfounded" --model $model --search_timeout 10 \
                --n_objects 10 --n_properties 10 --n_actions 0 --max_program_depth 6 \
                --seed $i --variable_distribution $distribution --max_imgs 10 --no_sampling

                CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "cocologic-max-img" --model $model --search_timeout 10 \
                --n_objects 10 --n_properties 10 --n_actions 3  --max_program_depth 6 \
                --seed $i --variable_distribution $distribution --max_imgs 10 --no_sampling


                CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "bongard-hoi-max-img" --model $model --search_timeout 10 \
                --n_objects 10 --n_properties 5 --n_actions 10 --max_program_depth 4 \
                --seed $i --variable_distribution $distribution --max_imgs 20 --no_sampling

                CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "CLEVR-Hans3-unconfounded" --model $model --search_timeout 10 \
                --n_objects 10 --n_properties 10 --n_actions 0 --max_program_depth 6 \
                --seed $i --variable_distribution $distribution --max_imgs 20 --no_sampling

                CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "cocologic-max-img" --model $model --search_timeout 10 \
                --n_objects 10 --n_properties 10 --n_actions 3  --max_program_depth 6 \
                --seed $i --variable_distribution $distribution --max_imgs 20 --no_sampling


                CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "bongard-hoi-max-img" --model $model --search_timeout 10 \
                --n_objects 10 --n_properties 5 --n_actions 10 --max_program_depth 4 \
                --seed $i --variable_distribution $distribution --max_imgs 30 --no_sampling

                CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "CLEVR-Hans3-unconfounded" --model $model --search_timeout 10 \
                --n_objects 10 --n_properties 10 --n_actions 0  --max_program_depth 6 \
                --seed $i --variable_distribution $distribution --max_imgs 30 --no_sampling

                CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "cocologic" --model $model --search_timeout 10 \
                --n_objects 10 --n_properties 10 --n_actions 3 --max_program_depth 6 \
                --seed $i --variable_distribution $distribution --max_imgs 30 --no_sampling


                CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "bongard-hoi-max-img" --model $model --search_timeout 10 \
                --n_objects 10 --n_properties 5 --n_actions 10 --max_program_depth 4 \
                --seed $i --variable_distribution $distribution --max_imgs 50 --no_sampling

                CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "CLEVR-Hans3-unconfounded" --model $model --search_timeout 10 \
                --n_objects 10 --n_properties 10 --n_actions 0  --max_program_depth 6 \
                --seed $i --variable_distribution $distribution --max_imgs 50 --no_sampling

                CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "cocologic" --model $model --search_timeout 10 \
                --n_objects 10 --n_properties 10 --n_actions 3 --max_program_depth 6 \
                --seed $i --variable_distribution $distribution --max_imgs 50 --no_sampling
            done
        done
    done
done