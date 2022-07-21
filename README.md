# ECCV-ABAW-competition

First, note that previous code has been stored in our 5 server abaw_4th/DAN/prev_coder

# Pretrain code
### mixaug code
```bash
   python mlt_finetuning_mixaug.py --aff_path1 /path/to/datasets/ -aff_path2 /path/to/landmark
```
### mixup code

```bash
   python mlt_finetuning_mixup.py --aff_path1 /path/to/datasets/ -aff_path2 /path/to/landmark
```

### test code 
```bash
   python test.py --testconfig test.txt --test_path /path/to/test/ --model /path/to/model/
```
### Our architectures, 
1. VGGFace2(ResNet50) + DAN(attention)

2. DINO(ResNet50) + DAN(attention)

3. MobileVitv2 + DAN(attention)


