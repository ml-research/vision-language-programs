
for i in 0 1 2
do
    for model in "InternVL3-8B" "InternVL3-14B" "Qwen2.5-VL-7B-Instruct" "Kimi-VL-A3B-Instruct"
    do

     CUDA_VISIBLE_DEVICES=$1 python baseline_structure.py --dataset "cocologic" --model $model  \
    --n_objects 10 --n_properties 10 --n_actions 3 \
    --seed $i --max_imgs 10

    CUDA_VISIBLE_DEVICES=$1 python baseline_structure.py --dataset "CLEVR-Hans3-unconfounded" --model $model \
    --n_objects 10 --n_properties 10 --n_actions 0  \
    --seed $i --max_imgs 10
    
    CUDA_VISIBLE_DEVICES=$1 python baseline_structure.py --dataset "bongard-op" --model $model  \
    --n_objects 10 --n_properties 10 --n_actions 3  \
    --seed $i --max_imgs 6

    CUDA_VISIBLE_DEVICES=$1 python baseline_structure.py --dataset "bongard-hoi" --model $model  \
    --n_objects 10 --n_properties 10 --n_actions 5  \
    --seed $i --max_imgs 6

    CUDA_VISIBLE_DEVICES=$1 python baseline_structure.py --dataset "bongard-hoi-max-img" --model $model  \
    --n_objects 10 --n_properties 10 --n_actions 5  \
    --seed $i --max_imgs 10

    done
done

