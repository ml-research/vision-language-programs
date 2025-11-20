

for model in "Qwen2.5-VL-7B-Instruct" "InternVL3-8B" "InternVL3-14B" "Qwen3-VL-30B-A3B-Instruct"
do
    for i in 0
    do

        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "CLEVR-Hans3-unconfounded" --model $model --seed $i --max_imgs 10
        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "cocologic-max-img" --model $model --seed $i --max_imgs 10 
        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "bongard-hoi-max-img" --model $model --seed $i --max_imgs 10

        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "CLEVR-Hans3-unconfounded" --model $model --seed $i --max_imgs 20 
        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "cocologic-max-img" --model $model --seed $i --max_imgs 20 
        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "bongard-hoi-max-img" --model $model --seed $i --max_imgs 20 

        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "CLEVR-Hans3-unconfounded" --model $model --seed $i --max_imgs 30 
        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "cocologic-max-img" --model $model --seed $i --max_imgs 30
        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "bongard-hoi-max-img" --model $model --seed $i --max_imgs 30 

        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "CLEVR-Hans3-unconfounded" --model $model --seed $i --max_imgs 50 
        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "cocologic-max-img" --model $model --seed $i --max_imgs 50
        CUDA_VISIBLE_DEVICES=$1 python baseline.py --dataset "bongard-hoi-max-img" --model $model --seed $i --max_imgs 50


    done
done
