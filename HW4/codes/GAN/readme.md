## Running Guide

```
python main.py --do_train --latent_dim 100
```

## Other Modifications

```
1. GAN\dataset.py实现了interpolate函数用于展示插值结果
2. 更改了dataloader以避免本地运行出错, num_workers=0, pin_memory=False
```

