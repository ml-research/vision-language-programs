

for i in 0 # 1 2
do
    # for model in "InternVL3-8B" "InternVL3-14B" "Qwen2.5-VL-7B-Instruct" "Qwen3-VL-30B-A3B-Instruct"
    # for model in "Qwen2.5-VL-7B-Instruct"
    for model in  "InternVL3-14B" "Kimi-VL-A3B-Instruct"
    # for model in "Ovis2.5-9B"
    do
    for distribution in "naive_weighted"
        do
        for max_program_depth in 5
            do

                # CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "bongard-hoi-max-img" --model $model --search_timeout 10 \
                # --n_objects 10 --n_properties 5 --n_actions 10 --max_program_depth 4 \
                # --seed $i --variable_distribution $distribution --max_imgs 10

                # --n_objects 10 --n_properties 10 --n_actions 0 --max_program_depth 6 \
                # --seed $i --variable_distribution $distribution --max_imgs 10

                # CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "cocologic-max-img" --model $model --search_timeout 10 \
                # --seed $i --variable_distribution $distribution --max_imgs 10

                # CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "bongard-hoi-max-img" --model $model --search_timeout 10 \
                # --n_objects 10 --n_properties 5 --n_actions 10 --max_program_depth 4 \
                # --seed $i --variable_distribution $distribution --max_imgs 20

                # CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "CLEVR-Hans3-unconfounded" --model $model --search_timeout 10 \
                # --n_objects 10 --n_properties 10 --n_actions 0 --max_program_depth 6 \
                # --seed $i --variable_distribution $distribution --max_imgs 20

                # CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "cocologic-max-img" --model $model --search_timeout 10 \
                # --n_objects 10 --n_properties 10 --n_actions 3  --max_program_depth 6 \
                # --seed $i --variable_distribution $distribution --max_imgs 20

                # CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "bongard-hoi-max-img" --model $model --search_timeout 10 \
                # --n_objects 10 --n_properties 5 --n_actions 10 --max_program_depth 4 \
                # --seed $i --variable_distribution $distribution --max_imgs 30

                # CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "CLEVR-Hans3-unconfounded" --model $model --search_timeout 10 \
                # --n_objects 10 --n_properties 10 --n_actions 0  --max_program_depth 6 \
                # --seed $i --variable_distribution $distribution --max_imgs 30

                # CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "cocologic-max-img" --model $model --search_timeout 10 \
                # --n_objects 10 --n_properties 10 --n_actions 3 --max_program_depth 6 \
                # --seed $i --variable_distribution $distribution --max_imgs 30

                CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "bongard-hoi-max-img" --model $model --search_timeout 10 \
                --n_objects 10 --n_properties 5 --n_actions 10 --max_program_depth 4 \
                --seed $i --variable_distribution $distribution --max_imgs 50

                # CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "CLEVR-Hans3-unconfounded" --model $model --search_timeout 10 \
                # --n_objects 10 --n_properties 10 --n_actions 0  --max_program_depth 6 \
                # --seed $i --variable_distribution $distribution --max_imgs 50

                # CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "cocologic-max-img" --model $model --search_timeout 10 \
                # --n_objects 10 --n_properties 10 --n_actions 3 --max_program_depth 6 \
                # --seed $i --variable_distribution $distribution --max_imgs 50

            done
        done
    done
done