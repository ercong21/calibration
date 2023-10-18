#!/bin/bash

REPO=$PWD
MODEL_NAME=${1:-"bert-base-cased"}
CALIBRATION_STRATEGY=${2:-"transform"}
GPU=${3:-0,1,2,3,4,5,6,7}
# TASK_NAMES=("ag_news" "amazon_polarity" "amazon_star" "xnli" "pawsx" "yahoo" "cola" "mrpc" "qnli" "qqp" "rte" "sst2" "wnli")
TASK_NAMES=("amazon_reviews_multi" "ag_news_multi" "pawsx_multi" "xnli_multi")
MAXL=128
EPOCH=0
LR=0.0001
# SEEDS=(1234 512 1213 421)
SEED=42
NUM_SAMPLES=(0)
BATCH_SIZE=8
LANGS='af,co,en,eo,haw,hmn,ht,ig,jw,km,mi,mn,mt,my,ny,or,sm,sn,st,sw,ta,te,tl,ug,ur,uz,zu'

export CUDA_VISIBLE_DEVICES=$GPU

if [ $MODEL_NAME == "bert-base-cased" ] || [ $MODEL_NAME == "bert-base-multilingual-cased" ]; then
    MODEL_TYPE='bert'
elif [ $MODEL_NAME == "roberta-base" ] || [ $MODEL_NAME == "xlm-roberta-base" ]; then
    MODEL_TYPE='roberta'
fi

RESULT_DIR="results_$MODEL_TYPE-$CALIBRATION_STRATEGY/"

run_task(){
    python run.py \
        --model_name $MODEL_NAME \
        --max_seq_length $MAXL \
        --task_name ${1} \
        --per_gpu_batch_size $BATCH_SIZE \
        --num_train_sample ${2} \
        --pattern_id ${3}\
        --penalty_train_epoch $EPOCH \
        --penalty_train_lr $LR \
        --seed ${4} \
        --save_train_logits \
        --result_dir $RESULT_DIR \
        --calibration_strategy $CALIBRATION_STRATEGY \
        --langs $LANGS \
        ${5} # save logits or penalize
}

# for SEED in "${SEEDS[@]}"
# do
for TASK in "${TASK_NAMES[@]}"
do
    if [ $TASK == "cola" ] || [ $TASK == "mrpc" ] || [ $TASK == "rte" ] || [ $TASK == 'sst2' ] || [ $TASK == 'pawsx' ] || [ $TASK == 'pawsx_multi' ]; then
        PATTERN=0
    elif [ $TASK == "amazon_polarity" ] || [ $TASK == "amazon_star" ] || [ $TASK == "xnli" ] || [ $TASK == "wnli" ]\
    [ $TASK == "xnli_multi" ] || [ $TASK == "amazon_reviews_multi" ] || [ $TASK == "qnli" ] || [ $TASK == 'qqp' ]; then
        PATTERN=1
    elif [ $TASK == "ag_news" ] || [ $TASK == "yahoo" ] || [ $TASK == "ag_news_multi" ]; then
        PATTERN=2
    fi

    if [ $TASK == 'xnli_multi' ]; then
        LANGS="ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh"
    elif [ $TASK == 'pawsx_multi' ]; then
        LANGS="de,en,es,fr,ja,ko,zh"
    elif [ $TASK == 'amazon_reviews_multi' ]; then
        LANGS="de,en,es,fr,ja,zh"
    elif [ $TASK == 'ag_news_multi' ]; then
        LANGS="af,co,en,eo,haw,hmn,ht,ig,jw,km,mi,mn,mt,my,ny,or,sm,sn,st,sw,ta,te,tl,ug,ur,uz,zu"
    fi

    NUM_SAMPLE=-1

    run_task $TASK $NUM_SAMPLE $PATTERN $SEED "--save_logits"

    for NUM_SAMPLE in "${NUM_SAMPLES[@]}"
    do
        run_task $TASK $NUM_SAMPLE $PATTERN $SEED "--penalize"
    done
done
# done