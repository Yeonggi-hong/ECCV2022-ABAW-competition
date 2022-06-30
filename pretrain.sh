#!/bin/bash
for num_class in 6 8 
do
data_path="/abaw_4th/3th_dataset/${num_class}_class/AFFECTNET_EXPW"
    for pretrained in "False" "True"
    do
        for model in  "DINO" "VGGFACE_DAN" "DINO_DAN" 
        do
            python pretrain.py --data_path ${data_path} --num_class ${num_class} --pretrained ${pretrained} --model ${model}
        done
    done
done