# roberta-base roberta-large microsoft/deberta-v3-base

python src/train_ft_glue.py \
  --task sst2 \
  --model roberta-large \ 
  --output-dir ./outputs/sst2-roberta-large \
  --epochs 3 --lr 2e-5 --train-bsz 32 --eval-bsz 64 --fp16
