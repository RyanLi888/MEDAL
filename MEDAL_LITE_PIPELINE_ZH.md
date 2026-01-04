# MEDAL-Lite 端到端流程说明（固定 lite4 + Dual-Stream Bi-Mamba）

> 本文档面向当前代码库的“新范式固定版”实现：
>
>- **输入特征固定为 lite4**：`[Length, Direction, BurstSize, ValidMask]`（维度=4）
>- **骨干网络固定为 DualStreamBiMambaBackbone**（双流双向 Mamba）
>- **Stage2 TabDDPM 固定在特征空间（feature-space）**生成 `(N, D)` 的合成特征

---

## 1. 符号与张量 Shape 约定

- `N`：样本数（flow 数）
- `L`：序列长度（固定 `1024`）
- `D_in`：输入特征维度（固定 `4`）
- `D`：骨干输出特征维度（通常 `OUTPUT_DIM`，例如 32）
- `X`：序列输入，shape `(N, L, D_in)`
- `Z`：骨干提取的特征向量，shape `(N, D)`

---

## 2. 全流程 Mermaid 总览图

```mermaid
flowchart TB
  subgraph S0[Stage 0: PCAP 解析与特征构造]
    A1[PCAP/PCAPNG 文件] --> A2[按 5 元组分流]
    A2 --> A3[逐包提取 lite4 特征\nLength, Direction, BurstSize, ValidMask]
    A3 --> A4[pad/truncate 到 L=1024\n得到 X: (N,1024,4)]
  end

  subgraph S1[Stage 1: Backbone 自监督预训练]
    B1[X: (B,1024,4)] --> B2[DualViewAugmentation]
    B2 --> B3[TrafficAugmentation\nCrop/Jitter/Mask]
    B3 --> B4[DualStreamBiMambaBackbone]
    B4 --> B5[SimMTM (可选 + Contrastive)]
    B5 --> B6[保存 backbone_pretrained.pth]
  end

  subgraph S2[Stage 2: 标签矫正 + 特征空间 TabDDPM]
    C1[X: (N,1024,4), y_noisy] --> C2[冻结 backbone 提取特征\nZ = backbone(X) -> (N,D)]
    C2 --> C3[HybridCourt 标签矫正\nkeep/drop/flip/reweight]
    C3 --> C4[TabDDPM(feature-space)\n训练 + 生成 Z_syn]
    C4 --> C5[保存 augmented_features.npz\nZ_aug: (N_aug,D)]
    C4 --> C6[保存 tabddpm_feature.pth]
  end

  subgraph S3[Stage 3: 分类器训练]
    D1[输入: Z_aug (N_aug,D)] --> D2[MEDAL_Classifier\n(跳过 backbone, 仅 dual_mlp)]
    D2 --> D3[DualStreamLoss\n加权监督/一致性/正交等]
    D3 --> D4[保存 classifier_best_f1.pth (full)]
    D3 --> D5[保存 classifier_final.pth (dual_mlp)]
    D3 --> D6[保存 model_metadata.json]
  end

  subgraph INF[Inference/Test]
    E1[X_test: (N,1024,4)] --> E2[z = backbone(X_test) -> (N,D)]
    E2 --> E3[dual_mlp(z) -> logits -> probs]
    E3 --> E4[阈值搜索 + 指标 + 可视化]
  end

  S0 --> S1
  S0 --> S2
  S1 --> S2
  S2 --> S3
  S1 --> INF
  S3 --> INF
```

---

## 3. Stage 0：PCAP → lite4 序列（`(N,1024,4)`）

文件：`MoudleCode/preprocessing/pcap_parser.py`

### 3.1 分流（Flow Split）
- **方法**：对每个包取 5 元组（`src_ip,dst_ip,src_port,dst_port,proto`）并做规范化（双向同一 flow key）

### 3.2 lite4 特征定义（逐包）
- **Length**：`min(len(pkt)/1500, 1.0)`
- **Direction**：相对首包方向，client→server 为 `+1`，反向为 `-1`
- **BurstSize**：
  - 只用 IAT（包间隔）做 burst 边界判定：方向变化或 `iat > BURST_IAT_THRESHOLD`
  - burst 内 raw length 求和，最后对每个包填充该 burst 的总和
  - 输出 `log1p(burst_sum)`
- **ValidMask**：真实包=1；padding 行=0（因为 padding 全 0）

### 3.3 序列对齐
- 截断：超过 1024 取前 1024
- padding：不足 1024 补全零行（含 ValidMask=0）

---

## 4. Stage 1：自监督预训练（DualStreamBiMambaBackbone + SimMTM）

入口：`scripts/training/train.py:stage1_pretrain_backbone`

### 4.1 双视图增强入口
- `DualViewAugmentation(config)` 内部调用 `TrafficAugmentation(config)`
- 对同一条序列生成两份视图：`x_view1, x_view2`

### 4.2 TrafficAugmentation 的三种增强
文件：`MoudleCode/feature_extraction/traffic_augmentation.py`

#### A) Asynchronous Temporal Cropping
- 概率：`AUG_CROP_PROB`
- 行为：随机截取连续片段放到序列开头，其余补 0

#### B) Inter-Arrival Jitter（抖动）——为什么会自动跳过？
位置：`TrafficAugmentation._temporal_jitter`

触发“跳过”的关键条件：
- `if self.iat_index is None: return x_jitter`

为什么会成立：
- `config.py` 固定 `IAT_INDEX = None`（lite4 没有 IAT / Log-IAT 维度）
- `TrafficAugmentation.__init__` 中读取：`self.iat_index = getattr(config, 'IAT_INDEX', None)`

因此：
- **在 lite4 下 jitter 的物理意义不存在**（没有 IAT 维度可加噪声）
- 所以实现上直接返回原序列，实现“自动跳过”

额外保护：
- 即使 iat_index 存在，也只会对 `ValidMask=1` 的 token 加噪声，避免污染 padding 区域。

#### C) Channel Masking（通道掩码）——什么时候会跳过？
位置：`TrafficAugmentation._channel_mask`

掩码维度选择：
- 只从 `[length_index, burst_index, cumulative_index]` 里选
- **不掩码** Direction/ValidMask（结构与 padding 语义不能破坏）

触发“跳过”的条件：
- 如果 `maskable_dims` 为空：`if len(maskable_dims) == 0: return x_mask`

为什么通常不会空：
- lite4 至少有 `Length(0)` 与 `BurstSize(2)` 可掩码

同样的 padding 保护：
- 如果存在 `VALID_MASK_INDEX`，仅对 `ValidMask=1` 的 token 掩码。

---

## 5. Stage 2：标签矫正 + 特征空间 TabDDPM

入口：`scripts/training/train.py:stage2_label_correction_and_augmentation`

### 5.1 冻结骨干提取特征
- 输入：`X_train (N,1024,4)`
- 输出：`Z = backbone(X_train, return_sequence=False)` → `(N,D)`

### 5.2 HybridCourt 标签矫正
- 输出 `keep_mask / y_corrected / correction_weight` 等
- 形成 clean 子集：`Z_clean, y_clean, weights_clean`

### 5.3 TabDDPM（feature-space）
文件：`MoudleCode/data_augmentation/tabddpm.py`

- 训练空间固定为特征向量空间
- 生成输出：`Z_augmented (N_aug,D)`

保存：
- `output/.../data_augmentation/models/augmented_features.npz`
- `output/.../data_augmentation/models/tabddpm_feature.pth`

> 注意：raw/sequence-level 的 `augment_dataset()` 已被硬禁用（直接 raise），防止误用。

---

## 6. Stage 3：分类器训练（特征输入为主）

入口：`scripts/training/train.py:stage3_finetune_classifier`

### 6.1 输入判定
- 若 `X_train.ndim == 2`，即 `(N,D)`：视为特征输入
  - 跳过 backbone
  - 关闭在线增强 / ST-Mixup

### 6.2 模型与输出
- `MEDAL_Classifier(backbone, config)` 内部：
  - 特征模式：`z = x`
  - 序列模式：`z = backbone(x)`
- `dual_mlp(z) -> logits -> probs`

保存：
- `classifier_best_f1.pth`：整模型 `classifier.state_dict()`
- `classifier_final.pth`：仅 `dual_mlp.state_dict()`
- `model_metadata.json`：记录训练时 backbone 路径等

---

## 7. 推理 / 测试

入口：`scripts/testing/test.py`

- 输入：`X_test (N,1024,4)`
- 前向：`logits, z = classifier(X_test, return_features=True)`
- 输出：`probs = softmax(logits)`

### 7.1 关于 best/final 两种 checkpoint
- 训练保存 best 与 final 的格式不同（full vs dual_mlp）
- 测试端已实现自适配加载（优先 full，失败回退 dual_mlp）

---

## 8. 快速核对清单（你排查问题时常用）

- `X` 的最后一维是否为 `4`？（必须）
- padding 行是否全 0？（必须，ValidMask=0）
- backbone pooling 是否使用 `ValidMask`？（是）
- jitter 是否在 lite4 下跳过？（是，IAT_INDEX=None）
- channel mask 是否只掩 Length/Burst？（是）
- Stage2 产物是否为 `augmented_features.npz`（Z 级别）？（是）
- test.py 能否加载 `classifier_final.pth`？（已支持）
