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

微调前:

human : 台湾有什么好吃的

bot: 台北有哪些值得一去的景点?

human :北京有啥好玩的地方

bot: <a href="http://www.bj-tourisme" target=_blank>北京旅游网</A><BR<br />

微调后：

human :  台湾有什么好吃的

bot: 1.蚵仔煎(Oyster Pancake)：用鸡蛋和面粉混合后摊成薄饼，再加入葱花、猪肉末等佐料。最后将生蚝肉放在上面即可食用;20世纪初由闽南移民传入台北市的传统小吃之一 ;3大肠包饭 (Da Dang Bao): 大米蒸熟后再加酱油调味并配以猪肚或鸡肝做成的馅儿,然后把这些食材包裹在糯香的大白菜叶子中一起煮制而成的一种美食 ,是当地人最爱的早餐食品！
       
human : 北京有啥好玩的地方 

bot: 1.故宫博物院：中国古代皇家建筑群，是明清两代的皇帝居住和办公场所。门票价格为60元/人次(淡季)、85 元 / 人 (旺季节假日期间);开放时间: 周二至周日9点-下午4点半;周一闭馆维护检修不对外售票,请提前购置参观券或预约入场方式进园游览观光拍照留念等活动需在指定区域内进行不得进入其他禁区及文物保护单位范围内游玩以免破坏古迹造成损失影响安全问题后果自负!2北京天坛公园位于北京市中心东城区永定门外大街南侧占地面积约73万平方米是中国现存最大的祭祀性园林之一也是世界上保存最完整的天文台遗址！免费景点无需购买任何费用但需要排队等待安保人员检查后方可通行建议游客最好早到避免长时间拥堵交通高峰期前往景区观赏美景感受历史文化魅力~
### Reference
https://github.com/artidoro/qlora
https://github.com/LianjiaTech/BELLE
