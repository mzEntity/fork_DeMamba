# DeMamba 使用说明

本文根据当前仓库代码整理，介绍环境准备、数据组织、训练命令和预处理入口。当前仓库没有提供依赖锁定文件，也没有包含完整数据集、预训练权重或 OSS 数据读取实现；首次运行前需要按下文准备这些资源。

## 1. 代码结构

- `train.py`：训练入口，通过 `--config` 指定 YAML 配置。
- `configs/XCLIP_DeMamba.yaml`：默认的 XCLIP+DeMamba 训练配置。
- `dataloader/GenVideo.py`：GenVideo 数据集适配器。
- `dataloader/AIGVDBench.py`：AIGVDBench 数据集适配器，直接读取 MP4 与 JSON 元信息。
- `dataloader/__init__.py`：根据配置中的 `dataset` 选择适配器。
- `models/DeMamba.py`：DeMamba 模型定义；XCLIP 编码器从本地目录加载。
- `Preprocess/video2frame.py`：视频抽帧脚本，当前脚本针对 `./SD` 目录编写。
- `Preprocess/folder2csv.py`：把 `./SD_frames` 下的帧目录生成 `SD.csv` 的示例脚本。

## 2. 环境准备

代码使用 PyTorch、CUDA、Transformers、OpenCV、Albumentations、Pandas、NumPy、scikit-learn、PyYAML 和 tqdm 等库。仓库没有 `requirements.txt` 或 `environment.yml`，请根据机器 CUDA 版本安装兼容的 PyTorch，再安装代码依赖。例如：

```bash
conda create -n demamba python=3.10 -y
conda activate demamba

# 请按本机 CUDA 版本选择 PyTorch 安装命令，参见 PyTorch 官方安装页面。
pip install transformers opencv-python albumentations pandas numpy scikit-learn pyyaml tqdm
pip install moviepy
```

预处理脚本需要系统安装 `ffmpeg`。训练脚本直接调用 `.cuda()` 并使用 `DataParallel`，因此默认要求可用的 NVIDIA GPU/CUDA 环境；目前代码不能直接在纯 CPU 环境训练。

> 以上是根据源码导入整理的基础依赖，不是经过验证的完整依赖锁定清单。不同版本之间可能需要进一步匹配。

## 3. 准备预训练权重和数据

### X-CLIP 权重

`models/DeMamba.py` 中的 `XCLIP_DeMamba` 从配置项 `pretrained_model_path` 载入视觉编码器。默认训练配置将它设为：

```text
GenVideo/pretrained_weights/xclip
```

请将兼容 `transformers.XCLIPVisionModel.from_pretrained()` 的模型文件放在该目录中。目录应包含 Transformers 能识别的配置和权重文件。

### 数据与 CSV

GenVideo 数据加载器当前固定读取以下 CSV 路径（相对于仓库根目录）：

```text
GenVideo/datasets/train.csv
GenVideo/datasets/val_id.csv
GenVideo/datasets/val_ood.csv
```

训练集始终读取 `train.csv`。验证集由配置里的 `task` 决定：默认 `many2many` 会读取 `val_ood.csv`；`normal` 会读取 `val_id.csv`；其他任务（包括配置中列出的攻击任务）读取相应 CSV，具体文件名见 `dataloader/GenVideo.py` 的 `generate_dataset_loader()`。自建数据集时，应将自己的训练/验证清单整理成这些文件名，或使用自己的数据集适配器。

#### 帧目录和路径规则

数据加载器不是直接用 CSV 的 `image_path` 打开图像，而是用 `GenVideo/` 加上 `content_path` 和帧编号拼接 OSS 对象键：

```text
GenVideo/<content_path>/<帧编号>.jpg
```

例如 `content_path` 为 `real/mydata/clip001`、`frame_seq` 含有 `[1, 2, 3]` 时，数据读取端会请求：

```text
GenVideo/real/mydata/clip001/1.jpg
GenVideo/real/mydata/clip001/2.jpg
GenVideo/real/mydata/clip001/3.jpg
```

`content_path` 应是相对标识，不要再包含对象前缀。每个视频的帧放在同名子目录中，文件名用帧编号加 `.jpg`；编号需要与 `frame_seq` 完全一致。帧数、输入尺寸和归一化参数由 `num_frames`、`image_size`、`image_mean`、`image_std` 控制；当前配置从每个样本取 8 帧并缩放到 `224×224`。

目录示例：

```text
GenVideo/
├── datasets/
│   ├── train.csv
│   └── val_ood.csv
└── real/
    └── mydata/
        └── clip001/
            ├── 1.jpg
            ├── 2.jpg
            └── 3.jpg
```

**但当前代码实际通过 `download_oss_file()` 读取上述逻辑路径。** 若数据只放在本机磁盘，还需要把该函数实现为本地路径读取，或替换数据加载调用；否则本地目录结构本身不会被自动读取。若继续使用 OSS，则 OSS 对象键需要与上面的拼接路径一致。

#### CSV 字段

训练集 `train.csv` 每行表示一个视频，至少需要：

| 字段 | 示例 | 说明 |
| --- | --- | --- |
| `content_path` | `real/mydata/clip001` | 视频标识和帧目录相对路径；也是结果中记录的数据路径 |
| `label` | `0` | 二分类标签：真实视频为 `0`，AI 生成视频为 `1` |
| `frame_seq` | `[1, 2, 3]` | Python 列表形式的帧编号字符串，读取时由 `eval()` 解析 |

验证 CSV（例如 `val_ood.csv`）需要以上字段，并且当前实现还会读取 `activity_id` 和 `image_path`：

| 字段 | 示例 | 说明 |
| --- | --- | --- |
| `activity_id` | `0` | 验证数据代码会读取该列，但当前评测流程没有用其值 |
| `image_path` | `GenVideo/real/mydata/clip001/1.jpg` | 初始化时会读取该列，但之后实际帧读取仍按 `content_path` 与 `frame_seq` 拼接 |

最小 CSV 示例（字段顺序不限，需有表头）：

```csv
content_path,label,frame_seq
real/mydata/clip001,0,"[1, 2, 3]"
fake/mygenerator/clip002,1,"[1, 2, 3, 4, 5, 6, 7, 8]"
```

验证 CSV 示例：

```csv
content_path,label,frame_seq,activity_id,image_path
real/mydata/clip101,0,"[1, 2, 3]",0,GenVideo/real/mydata/clip101/1.jpg
fake/mygenerator/clip102,1,"[1, 2, 3, 4, 5, 6, 7, 8]",0,GenVideo/fake/mygenerator/clip102/1.jpg
```

CSV 标签和目录名之间没有自动校验；请确保 `label` 与样本真实/生成类别一致。可以增加来源、生成器名称等其它列，但当前代码不会自动利用它们。`content_path` 的命名最好保留来源层级（如 `fake/<生成器>/...`），这样评测结果里仍可辨认数据来源。

### 当前仓库的数据读取限制

`dataloader/GenVideo.py` 目前调用 `download_oss_file()`，但仓库中没有定义或导入该函数。即使 CSV、对象前缀和图像路径已配置，GenVideo 适配器仍需项目环境提供 OSS 下载函数，或将读取逻辑改成本地文件读取。该限制不影响 AIGVDBench 适配器，它直接用 OpenCV 打开 MP4 文件。仓库 README 提到的 GenVideo 数据集及外部权重仍需自行准备。

## 4. 预处理（可选）

`Preprocess/video2frame.py` 面向脚本内写定的 `./SD` 输入目录和 `./SD_frames` 输出目录，安装 `ffmpeg` 后执行：

```bash
python Preprocess/video2frame.py
```

`Preprocess/folder2csv.py` 扫描帧目录并写出 `SD.csv`：

```bash
python Preprocess/folder2csv.py
```

注意：`folder2csv.py` 中的样例内容路径仍有硬编码，需要按实际数据位置调整。它生成的 CSV 不会自动复制到 GenVideo 训练清单位置。

## 5. 启动训练

在仓库根目录执行：

```bash
python train.py --config configs/XCLIP_DeMamba.yaml
```

程序读取配置并构造 `XCLIP_DeMamba`，然后开始训练和验证。默认参数位于 `configs/XCLIP_DeMamba.yaml`：

- `max_epoch: 30`：训练轮数。
- `train_batch_size: 128`、`val_batch_size: 64`：训练和验证批大小。
- `num_workers: 64`：数据加载进程数；可按 CPU/内存调整。
- `lr: 0.000001`：学习率。
- `save_dir: ./results/XCLIP_Mamba_2`：权重和评估结果输出目录。
- `task: many2many`：验证清单任务选择；默认会用 `val_ood.csv`。
- `tuning_mode: sft`：当前训练入口仅在值为 `lp` 时冻结 `model.encoder`，其它值会训练全部参数。

训练过程中会在 `save_dir` 写入 `last.pth`、验证准确率最高时的 `best_acc.pth`，以及每轮的准确率/AUC/AP 结果文本。修改配置后可通过同一命令指定该 YAML 文件。训练脚本会检查必需字段；缺少字段时直接报出错误，不会从代码中补默认值。

## 6. 接入其它数据集：数据提供器接口

如果新数据集的底层组织方式不同，不必强行转换成 GenVideo 的磁盘目录或 CSV。可以在 `dataloader/` 下为其编写一个模块，并通过包入口选择；关键是让它返回的两个 `DataLoader` 保持以下 batch 接口。训练代码依赖的是 batch 结构，不是原始数据集的文件格式。

### 训练 DataLoader

训练循环按顺序解包四个元素：

```python
index, video, label_onehot, binary_label
```

每个 batch 的形状/类型应为：

| 元素 | 每批形状 | 类型与含义 |
| --- | --- | --- |
| `index` | `[B]` 或等价可忽略标识 | 样本编号；训练主体不使用其内容 |
| `video` | `[B, 1, T, 3, 224, 224]` | 浮点图像张量；代码随后执行 `video[:, 0]`，所以去掉第二维后必须是 `[B, T, 3, H, W]` |
| `label_onehot` | `[B, 2]` | 二分类 one-hot 标签，类别顺序 `[真实, 生成]`，即真实 `[1, 0]`、生成 `[0, 1]` |
| `binary_label` | `[B, 1]` | 浮点二分类标签，真实为 `0`，生成为 `1`，用于 `BCEWithLogitsLoss` |

模型收到的输入为 `[B, T, 3, 224, 224]`，输出 logits `[B, 1]`。默认模型配置按 8 帧设计；自定义 `T` 时需确认模型对时间长度的处理兼容。

### 验证 DataLoader

验证循环按顺序解包五个元素：

```python
index, video, label_onehot, binary_label, video_id
```

前四项形状和含义与训练 batch 相同。`video_id` 是每个样本可读的字符串标识（如来源/文件名），用于评测结果中的 `data_path` 列；验证代码会将各批次的 ID 汇总。建议每个样本返回一个字符串，批处理后是一批字符串序列。

简化的样本返回逻辑示意：

```python
# video_frames: [T, 3, 224, 224] 的 float Tensor
# label: 0=真实，1=生成
video = video_frames.unsqueeze(0)  # [1, T, 3, 224, 224]
label_onehot = torch.tensor([1.0, 0.0]) if label == 0 else torch.tensor([0.0, 1.0])
binary_label = torch.tensor([float(label)])

# 训练样本
return index, video, label_onehot, binary_label

# 验证样本额外返回稳定、可读的样本 ID
return index, video, label_onehot, binary_label, video_id
```

`DataLoader` 自动添加 batch 维，因此单样本的 `video` 应为 `[1,T,3,224,224]`，组 batch 后为 `[B,1,T,3,224,224]`。如果希望把视频存成其它格式、从数据库或流式源读取，只需在自定义 `Dataset.__getitem__()` 中完成读取和预处理，再返回上述张量/标签即可。也可以在 `generate_dataset_loader()` 中根据配置选择自定义训练集和验证集，并仍然返回 `(train_loader, val_loader)`。

此外，当前 `eval_model()` 会用 `label_onehot` 计算分类指标，并将 `binary_label` 传给 BCE 损失；因此两种标签都要返回且保持一致。验证集每个类别都应有样本，评估还会计算 AUC/AP。

## 7. AIGVDBench 运行

仓库已提供 `dataloader/AIGVDBench.py`，直接从 MP4 读取视频，并使用元信息 JSON 中的标签。数据和元信息路径由配置项 `aigvdbench_data_root` 与 `aigvdbench_meta_root` 指定；当前配置指向：

```text
/root/misk1024/AIGVDBench
/root/misk1024/AIGVDBench_meta
```

其预期目录组织是 `AIGVDBench/<split>/<real|fake>/<来源>/<视频>.mp4`，元信息是 `AIGVDBench_meta/<split>/<real|fake>/<来源>.json`。适配器按 `train` split 训练、`val` split 验证；`test` split 当前不参与训练入口的验证。当前训练配置只选 `real/real` 和 `fake/T2V_Open-Sora` 两个训练 source。两类各 14,000 个样本，训练集共 28,000 个视频，每个 epoch 会遍历全部样本，最后一个不足整批的 batch 也会保留。验证也只使用 `real/real` 和 `fake/T2V_Open-Sora`。训练视频随机取连续 8 帧，验证视频均匀取 8 帧，缩放到 `224×224` 并按 ImageNet 均值/方差归一化。

使用单独配置启动：

```bash
python train.py --config configs/XCLIP_DeMamba_AIGVDBench.yaml
```

若数据放在其它位置，修改配置中的 `aigvdbench_data_root` 和 `aigvdbench_meta_root`。训练/验证 source 分别由 `aigvdbench_train_sources` 和 `aigvdbench_val_sources` 控制，权重位置由 `pretrained_model_path` 控制。训练配置将 DeMamba 原有训练参数保留，epoch 数设为 10；这些路径和 source 配置缺失时会直接报错。每轮结束后会保存 `Epoch_N.pth`、`Epoch_N_accuracy.txt`；同时更新 `last.pth`，并在验证准确率刷新时更新 `best_acc.pth`。

## 8. 常见运行问题

1. **找不到模型权重**：检查配置里的 `pretrained_model_path` 是否指向有效的本地权重目录或 Hugging Face 模型 ID。
2. **找不到 GenVideo CSV 或帧文件**：确认从仓库根目录启动，并检查 `GenVideo/datasets/` 下对应任务的 CSV 与数据路径。
3. **`download_oss_file` 未定义**：需补充 OSS 访问实现或替换为本地读取实现；此功能当前不在仓库中。
4. **CUDA 初始化失败**：当前训练代码无 CPU 回退，确认安装了匹配的 CUDA 版 PyTorch 且 GPU 可用。
5. **显存或内存不足**：降低 YAML 中的 batch size 和 `num_workers`。

## 9. 论文与数据集

项目 README 提供论文引用、GenVideo 数据集下载说明及项目背景。数据集本身不包含在本仓库中，使用前请按 README 中的来源和许可获取数据。
