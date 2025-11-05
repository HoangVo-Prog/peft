# peft

| Model | Mode | CoLA | SST | MRPC | QQP | STS | MNLI | QNLI | RTE | WNLI |
|:------|:------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| **roberta-base** | Finetune<br>n_params: 125M | 0.5880<br>~ | 0.9415<br>~ | 0.8725 / 0.9088<br>~ | 0.9140 / 0.8854<br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ |
|  | LoRA<br>n_params: 1.2M (trainable) | <br>~ | 0.9300<br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ |
|  | QLoRA<br>n_params: 1.2M (trainable, 4-bit) | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ |
| **roberta-large** | Finetune<br>n_params: 355M | 0.6258<br>~ | 0.9610<br>~ | 0.8946 / 0.9260<br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ |
|  | LoRA<br>n_params: 3.2M (trainable) | <br>~ | 0.9530<br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ |
|  | QLoRA<br>n_params: 3.2M (trainable, 4-bit) | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ |
| **deberta-v3-base** | Finetune<br>n_params: 183M | 0.6700<br>~ | 0.9576<br>~ | 0.8873 / 0.9179<br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ |
|  | LoRA<br>n_params: 1.8M (trainable) | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ |
|  | QLoRA<br>n_params: 1.8M (trainable, 4-bit) | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ | <br>~ |
