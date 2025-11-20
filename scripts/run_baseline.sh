

for model in "InternVL3-8B" "InternVL3-14B" "Qwen2.5-VL-7B-Instruct" "Kimi-VL-A3B-Instruct" # "Qwen3-VL-30B-A3B-Instruct" "Qwen3-VL-30B-A3B-Thinking" "gpt-5"
do
    for i in 0 1 2
    do
        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "CLEVR-Hans3-unconfounded" --model $model --seed $i --max_imgs 10 
        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "cocologic" --model $model --seed $i --max_imgs 10 
        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "bongard-op" --model $model --seed $i --max_imgs 6
        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "bongard-hoi" --model $model --seed $i --max_imgs 6
        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "bongard-rwr" --model $model --seed $i --max_imgs 6
    done
done
