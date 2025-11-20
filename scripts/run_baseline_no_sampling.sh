

for model in "InternVL3-8B" "InternVL3-14B" "Qwen2.5-VL-7B-Instruct" "Kimi-VL-A3B-Instruct"
do
    for i in 0
    do
        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "CLEVR-Hans3-unconfounded" --model $model --seed $i --max_imgs 10 --no_sampling
        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "cocologic" --model $model --seed $i --max_imgs 10 --no_sampling
        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "bongard-rwr" --model $model --seed $i --max_imgs 6 --no_sampling
        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "bongard-op" --model $model --seed $i --max_imgs 6 --no_sampling
        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "bongard-hoi" --model $model --seed $i --max_imgs 6 --no_sampling
   done
done

