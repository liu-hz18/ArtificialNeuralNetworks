## Running Guide

```
python main.py --do_train --latent_dim 100
```

## Other Modifications

```
1. VAE\dataset.py实现了interpolate函数用于展示插值结果
2. 更改了dataloader以避免本地运行出错, num_workers=0, pin_memory=False
3. VAE\VAE.py中，为了简洁VAE Net初始化过程，封装了Reshape层
```

