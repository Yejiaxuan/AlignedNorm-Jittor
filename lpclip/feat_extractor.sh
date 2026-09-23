# sh feat_extractor.sh
DATA="${DATA_ROOT:-/path/to/datasets}"
OUTPUT='./clip_feat/'
SEED=1
WEIGHTS="${JITTOR_RN50_WEIGHTS:?Set JITTOR_RN50_WEIGHTS to converted RN50 weights}"

# oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet
for DATASET in oxford_pets
do
    for SPLIT in train val test
    do
        python feat_extractor.py \
        --split ${SPLIT} \
        --root ${DATA} \
        --seed ${SEED} \
        --dataset-config-file ../configs/datasets/${DATASET}.yaml \
        --clip-weights ${WEIGHTS} \
        --output-dir ${OUTPUT} \
        --eval-only
    done
done
