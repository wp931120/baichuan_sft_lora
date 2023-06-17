# baichuan_sft_lora
baichuan LLM supervised finetune by lora

### 大模型
百川7B 
https://huggingface.co/baichuan-inc/baichuan-7B

### sft 数据集
采用的是belle 0.5M
https://huggingface.co/datasets/BelleGroup/train_0.5M_CN

### 训练方法和过程可视化
+ 先将百川LLM 采用qlora的 nf4 和双重量化方式进行量化
+ 在采用lora进行指令微调
+ 训练过程采用tensorborad 可视化,执行下方代码即可在localhost:6006去监控你的训练和测试loss
tensorboard  --logdir ./runs/ --bind_all


![image](https://github.com/wp931120/baichuan_sft_lora/assets/28627216/8a0cece1-189a-42a1-ab38-79f244e95d06)
![image](https://github.com/wp931120/baichuan_sft_lora/assets/28627216/0f4897a1-cc9b-440d-a962-dfe5e3da8711)



### 资源消耗
由于采用了int4量化和lora等技术
整个资源消耗只需要12G左右的显存

### sft后的效果
TODO
### Reference
https://github.com/artidoro/qlora
https://github.com/LianjiaTech/BELLE
