# Introduction to Language Model
通过在金庸先生的小说《笑傲江湖》训练简单的二元语言模型和GPT，帮助读者了解语言模型。项目非常的简单，每个语言模型包括了定义超参数、构造词表、处理训练数据、定义模型、训练模型、测试。每个部分都采用了非常简单的实现方式。代码有注释、可读性强。

## 环境


```
pkuseg # 分词器
torch 
```
## 二元语言模型

```python bigram.py```


## GPT 
```python miniGPT.py```

## 例子
* 训练前
   ```
   令狐冲冒气娶妻飞起狭隘露乖开窗酒楼有所不为病夫请客剖析分毫伯取圆石自失冲正眼梳子走惯江无知无识大害奠定何月咕抚迎刃而解收殓林何方老拳师延缓心腹大患泪光薄幸揭挚爱瞧罢山思正行挑柴通报马虎坚甚行攻似直饥渴甚众毛贼自由自在吁吁兄率开革心中一凛她不见天日灰影嚼食万万语塞三击青布巨利卷轴从不以刀作剑高人一等受尽......
   ```
    可以看到，模型的输出完全是随机的。

* 训练过程
  ```
  step: 0 | train loss: 10.7411 | val loss: 10.7421 | time: 0.0 min
  step: 1000 | train loss: 5.5786 | val loss: 6.0436 | time: 0.3 min
  step: 2000 | train loss: 5.1275 | val loss: 5.9307 | time: 0.6 min
  step: 3000 | train loss: 4.8512 | val loss: 5.9628 | time: 0.9 min
  step: 4000 | train loss: 4.6834 | val loss: 6.0009 | time: 1.1 min
  step: 5000 | train loss: 4.5729 | val loss: 6.0709 | time: 1.4 min
  step: 6000 | train loss: 4.4768 | val loss: 6.0861 | time: 1.7 min
  step: 7000 | train loss: 4.4103 | val loss: 6.1138 | time: 2.0 min
  step: 8000 | train loss: 4.3593 | val loss: 6.1525 | time: 2.3 min
  step: 9000 | train loss: 4.3136 | val loss: 6.1668 | time: 2.6 min
  ```
  损失下降，在慢慢收敛。训练后的输出如下，内容经不起检验，但是出现了吸星大法.....,至少有个样子了。

  ```
  令狐冲见到岳夫人叹了。岳灵珊的是谁说令狐冲不开，四名汉子哼，铁链缠住那雄伟的大路上红缨抖开一的道人打开了一挥，这时不转睛的恶徒，折而用指甲，只见那不是真的有各刺入。余沧海跃，道：“他潜运竟甘舍己命“正月人，赶车一般已给你的众人，抢出这几下以西掌门极未至茶，高举，原来是武林门户力道贯到右臂的洪福。霎时之间，登时压倒谢倒欢呼：“是谁，欺到一用兵刃，大笔淋漓，出剑 都已是个光明磊落的老头子道：“吸星大法”王家骏道：“我之意。”盈盈在鼓里。岳灵珊......
  ```