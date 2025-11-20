

for i in 0 1 2
do
    for distribution in "naive_weighted" # "naive_frequency" 
        do
        for max_program_depth in 5
            do
            for max_imgs in 6
            do
                # CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "cocologic" --model $2 --search_timeout 10 \
                # --n_objects 10 --n_properties 10 --n_actions 3  --max_program_depth 5 \
                # --seed $i --variable_distribution $distribution --max_imgs 10

                # CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "bongard-op" --model $2 --search_timeout 10 \
                # --n_objects 10 --n_properties 10 --n_actions 3 --max_program_depth 4 \
                # --seed $i --variable_distribution $distribution --max_imgs $max_imgs

                # CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "bongard-hoi" --model $2 --search_timeout 10 \
                # --n_objects 10 --n_properties 5 --n_actions 10  --max_program_depth 4 \
                # --seed $i --variable_distribution $distribution --max_imgs $max_imgs

                # CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "CLEVR-Hans3-unconfounded" --model $2 --search_timeout 10 \
                # --n_objects 10 --n_properties 10 --n_actions 0  --max_program_depth 5 \
                # --seed $i --variable_distribution $distribution --max_imgs 10

                CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "bongard-rwr" --model $2 --search_timeout 10 \
                --n_objects 10 --n_properties 10 --n_actions 5  --max_program_depth 4 \
                 --seed $i --variable_distribution $distribution --max_imgs $max_imgs

            done
        done
    done
done
