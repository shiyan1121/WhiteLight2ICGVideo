# ConvGRU–ResUNet 核心代码

整理日期：2026-11-23。默认对应主实验 `261011-ablation_261009_no_time`，不是时间条件版本。

## 范围与结构

只保留 ConvGRU–ResUNet 及训练、推理所需模块，不包含其他生成基线、Population Template、实验结果图和大体积 checkpoint；原桌面代码和原实验目录均不修改。

模型以静态 RGB 和血管掩膜为条件，通过残差 U-Net 编码器、瓶颈 ConvGRU、带跳跃连接的解码器生成32帧序列。默认通道宽度为32/64/128/256/512。辅助血管分支预测血管概率图，其特征经 `tanh` 门控后对荧光分支进行残差乘性调制。保留原参数名、计算和checkpoint兼容性，不改变历史模型行为。

默认 `include_time: false`、`previous_frames: 0`：不输入显式时间或前一帧ICG，时间变化由循环隐状态产生。代码中的可选时间/历史输入帮助函数仅为兼容同一模型家族，默认不启用。`nnunet` / `ConvGRUNNUNet` 只是旧配置/类名兼容别名；此处网络称为残差U-Net，不依赖 nnU-Net 软件。

```text
Code/
  configs/convgru_resunet.yaml    主实验训练配置（新的输出目录）
  train.py                      训练、验证、模型选择与断点续训
  predict.py                    RGB＋RGB来源VM的独立推理，不读取GT
  infer_patch.py                原数据组织下的批量推理与评价
  smoke_test.py                 无临床数据/预训练权重的快速连通性测试
  wlicgv/models.py              残差块、ConvGRU、门控双输出头
  wlicgv/data.py                配对patch、掩膜、帧索引/时间读取
  wlicgv/losses.py              重建、结构、辅助血管等损失
  wlicgv/time_templates.py       原推理接口所需时间模板工具
  wlicgv/utils.py               患者划分、日志与预览
  eval/                         原项目PSNR/SSIM/LPIPS/FVD兼容实现
  notebook/                     原split级指标汇总脚本
  scripts/run_train.bash         训练启动脚本（Git Bash）
  scripts/run_inference.bash     测试集best RGB2Vessel mask推理与汇总
```

## 1. 环境与快速检查

在本目录打开 PowerShell / Git Bash，建议 Python 3.10 或更高版本，并根据硬件安装合适的 PyTorch。

```powershell
python -m pip install -r requirements.txt
python smoke_test.py
```

快速检查用缩小的网络运行真实训练损失、反向传播、AdamW更新、32步双输出展开及checkpoint保存/严格重载。它验证代码连通性，不代表重训论文模型。依赖文件声明最低需求，不是原训练环境锁定文件。

## 2. 训练

默认配置来自主实验，保留原患者划分、32帧、256×256、seed 2026、60 epochs、AdamW（lr=2e-4、betas=(0.5,0.999)、weight_decay=0）以及验证SSIM最大化选择best的规则。损失权重沿用原实验，不额外增加论文示意图之外或未启用的损失。

先修改 `configs/convgru_resunet.yaml` 中 `data.root` / `data.rgb2vessel_root`（若更换机器）。`output.root: .` 将新结果写入本目录的 `experiments/日期-convgru_resunet_no_time`，不覆盖旧主实验。

```powershell
python train.py --config configs/convgru_resunet.yaml
```

或 Git Bash：`bash scripts/run_train.bash`。恢复训练：

```powershell
python train.py --config configs/convgru_resunet.yaml --resume "新实验目录/checkpoints/latest.pt"
```

训练期 `input_vessel_mask_source: global`，`guidance_vessel_mask_source: global`，与原主实验一致；逐帧辅助监督仍从原数据的关键帧血管掩膜读取。推理脚本显式切换到RGB来源二值掩膜。不要把global结构输入与逐帧辅助监督混为一谈。

原数据布局示意（保留原命名和manifest）：

```text
data_root/S23-260519_09/
  rgb_keyframe_patches/baseline_frame.png
  S23-260519_09_vessel_mask.png
  icg_keyframe_patches/Intensity_step_sample/...
  vessel_mask_keyframe_patches/Intensity_step_sample/...
RGB2Vessel/masks/S23-260519_09_baseline_frame.png
```

## 3. 仅用RGB与VM生成序列（不需要目标ICG）

`predict.py`只需可信的no-time checkpoint、RGB和事先计算的RGB2Vessel掩膜；不训练RGB2Vessel分割器，也不从GT提取输入。默认生成32张荧光PNG、32张辅助掩膜PNG和一份来源记录。概率图改用 `--mask-kind probability`。输出目录必须为空，避免覆盖。

```powershell
python predict.py --checkpoint "E:/ClinicalICG/WhiteLight2ICGVideo/code/4-WhiteLight2ICGVideo-ConvGRU/experiments/261011-ablation_261009_no_time/checkpoints/best.pt" --rgb "E:/ClinicalICG/WhiteLight2ICGVideo/data/3-KeyFramesVideo/S23-260519_09/rgb_keyframe_patches/baseline_frame.png" --mask "E:/ClinicalICG/WhiteLight2ICGVideo/data/RGB2Vessel/masks/S23-260519_09_baseline_frame.png" --out-dir "outputs/S23-260519_09"
```

输出位置是循环展开步，不是等间隔的真实采集时间；PNG是8-bit显示结果。该入口不执行GT指标计算，不需要LPIPS/FVD权重。

## 4. 原测试集批量推理与评价

完整指标额外需要：

```powershell
python -m pip install -r requirements-eval.txt
```

LPIPS及FVD特征网络可能需要下载权重或使用现有缓存。Git Bash中，默认引用原主实验checkpoint和数据，只在新Code目录写入推理结果：

```bash
bash scripts/run_inference.bash
```

指定新的已训练实验目录：

```bash
bash scripts/run_inference.bash "./experiments/日期-convgru_resunet_no_time"
```

只推理 `test`、`best`、`rgb2vessel_mask`，随后执行split级汇总。GT只用于该评价入口，不是预测输入；本no-time模型不使用case或train_mean时间作为网络输入。

`eval/cal_fvd.py`保留旧项目逐patch FVD接口，仅用于历史兼容，不等同于独立的cohort-level FVD；不要据此宣称完成了cohort FVD或subject级统计。本代码包不包含完整论文血流动力学分析流程。

## 来源与验证

来源为桌面 `RGB2ICGVideo结果整理/2-实验代码及结果/Code` 中的代码快照和主实验配置。网络文件已移除普通U-Net、Attention U-Net、U-Net++、pix2pixHD及其判别器等独立模型。主模型的辅助门控数学计算和state_dict键名保持不变。

具体校验记录见 `VERIFICATION.json`。未重新执行60轮完整训练，也未改写或复制原实验checkpoint、原始数据和已有结果。
