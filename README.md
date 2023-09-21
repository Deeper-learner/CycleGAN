## **CycleGAN-Pytorch**

- 可以用于图像风格迁移，去雾等应用

## **环境配置**

- **Python 3.8**
- **Pytorch 1.11.0**
- **Windows**

## **文件结构**

```
├── models: 网络结构定义，包括生成器和判别器
├── utils: 预处理模块，也包括缓冲池配置
├── imgs: 数据集的存放位置，其中包含content和style两个文件夹
├── train.py: 训练入口
```

## 数据集

```
├── imgs
	├── content
	├── style

其中content和style文件夹中分别存放原数据图片和风格数据图片
```

## 训练方法

由于缓冲池的存在，训练的batch_size只能为1，若需要增大batch_size，可以去掉缓冲池