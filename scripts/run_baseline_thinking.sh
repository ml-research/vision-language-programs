

for model in "Qwen3-VL-30B-A3B-Thinking" "gpt-5" # "gpt-4o"
# for model in "Kimi-VL-A3B-Thinking-2506" #"Qwen3-VL-30B-A3B-Thinking"
do
    for i in 0
    do
        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "CLEVR-Hans3-unconfounded" --model $model --seed $i --max_imgs 10 --think
        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "cocologic" --model $model --seed $i --max_imgs 10 --think

    done
done

