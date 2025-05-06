feature_dir=clip-georsclip_feat

for DATASET in EuroSAT
do
    python linear_probe.py \
    --dataset ${DATASET} \
    --feature_dir ${feature_dir} \
    --num_step 8 \
    --num_run 3
done
