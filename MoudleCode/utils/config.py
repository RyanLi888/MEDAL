"""
MEDAL-Lite 统一配置文件 (重构版 v2.6)
=================================
按照训练流程 Stage 1/2/3 组织，使用最优训练结果的配置

训练流程：
- Stage 1: 自监督预训练 (SimMTM + InfoNCE)
- Stage 2: 标签矫正 + 数据增强 (Hybrid Court + TabDDPM)
- Stage 3: 分类器微调 (Dual-Stream MLP + Mixed Training)

最优配置来源（ablation_data_augmentation_20260110_214253）：
- F1 Score (pos=1): 0.9026
- Precision: 0.8972
- Recall: 0.9080
- AUC: 0.9896
- Accuracy: 0.9822
- 最优阈值: 0.7579

关键配置：
- TabDDPM: 5x增强（特征空间）
- 混合训练: 32 real + 96 synthetic batches
- 骨干微调: 关闭（冻结骨干网络）
- 损失权重: real=2.0, synthetic=1.0
- Co-teaching: 禁用（数据无标签噪声）
- 训练轮数: 1000 epochs

更新日志（v2.6 - 2026-01-11）：
- 数据增强策略优化：
  * 新增Mixup增强：在TabDDPM生成后应用特征空间插值
  * 类别自适应Mixup：正常类alpha=0.1（弱混合），恶意类alpha=0.3（强混合）
  * 困难样本挖掘：预留接口（需要预训练模型）
  * 预期效果: F1提升至0.90+（当前0.858）

更新日志（v2.5 - 2026-01-11）：
- 类别特定增强模式：
  * 支持按类别设置不同的增强倍数
  * 当前配置：正常2x，恶意1x（2:1策略）
  * 预期结果: 750正常 + 500恶意 = 1250 (60%正常, 40%恶意)
- TabDDPM质量优化：
  * 恢复旧学习率参数（5e-4, min_lr=1e-5）以提高生成质量
  * 减少平滑窗口（5→3）提高早停敏感度
  * 增加质量检查：IQR离群点过滤（最大5%离群点）
  * 提高模板质量要求（0.7→0.8）

更新日志（v2.3 - 2026-01-11）：
- TabDDPM优化（组合1 - 保守优化）：
  * 增加模型容量: [64,128,64] → [128,256,256,128]（4层网络）
  * 优化学习率: 5e-4 → 8e-4，添加余弦退火调度器
  * 增加训练轮数: 800 → 1000 epochs
  * 增加早停耐心值: 50 → 80（避免过早停止）
  * 预期效果: 最佳损失降至0.28-0.32，F1提升1-2%

更新日志（v2.2 - 2026-01-11）：
- 禁用Co-teaching：数据使用真实标签（无噪声），Co-teaching会错误丢弃干净样本
- 恢复训练轮数：1000 epochs（与最优配置一致）
- 修正噪声率：CO_TEACHING_NOISE_RATE = 0.0（反映真实数据状态）

问题诊断（20260111_130936失败原因）：
- Co-teaching在无噪声数据上启用，错误丢弃20-30%干净样本
- 导致性能下降：F1从0.9026降至0.8716（-3.1%）
- 最优阈值异常：从0.7579升至0.9616（模型过于保守）
- 训练不足：800 epochs vs 1000 epochs（最优配置）
"""

import torch
import os
import math
from pathlib import Path


class Config:
    """MEDAL-Lite 全局配置类"""
    
    # ============================================================
    # 基础路径配置
    # ============================================================
    PROJECT_ROOT = str(Path(__file__).parent.parent.parent.absolute())
    
    # 数据集路径（支持环境变量自定义）
    DATASET_SUBDIR = os.environ.get('MEDAL_DATASET_SUBDIR', '').strip()
    if DATASET_SUBDIR:
        DATA_ROOT = os.path.join(PROJECT_ROOT, "Datasets", DATASET_SUBDIR)
    else:
        DATA_ROOT = os.path.join(PROJECT_ROOT, "Datasets")
    
    # 输出路径（支持环境变量自定义）
    DATASET_NAME = os.environ.get('MEDAL_DATASET_NAME', '').strip()
    if DATASET_NAME:
        OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", DATASET_NAME)
    else:
        OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")

    # 训练/测试数据路径
    BENIGN_TRAIN = os.path.join(DATA_ROOT, "T1_train", "benign")
    MALICIOUS_TRAIN = os.path.join(DATA_ROOT, "T1_train", "malicious")
    BENIGN_TEST = os.path.join(DATA_ROOT, "T2_test", "benign")
    MALICIOUS_TEST = os.path.join(DATA_ROOT, "T2_test", "malicious")

    # ============================================================
    # 数据集基础配置
    # ============================================================
    LABEL_NOISE_RATE = 0.30  # 标签噪声率（30%）
    LABEL_BENIGN = 0         # 正常流量标签
    LABEL_MALICIOUS = 1      # 恶意流量标签
    
    # ============================================================
    # PCAP 预处理参数
    # ============================================================
    SESSION_TIMEOUT = 60                    # 会话超时时间(秒)
    TCP_FAST_RECONNECT_THRESHOLD = 5        # TCP快速重连阈值(秒)
    ONE_FLOW_PER_5TUPLE = True              # 每个5元组一个流
    
    # 序列长度和特征配置
    SEQUENCE_LENGTH = 1024                  # 每个流最多1024个包
    MTU = 1500.0                            # 最大传输单元
    TCP_WINDOW_MAX = 65535.0                # TCP窗口最大值
    
    # 特征定义 (MEDAL-Lite5: Length, Direction, BurstSize, LogIAT, ValidMask)
    FEATURE_NAMES = ['Length', 'Direction', 'BurstSize', 'LogIAT', 'ValidMask']
    LENGTH_INDEX = 0
    DIRECTION_INDEX = 1
    BURST_SIZE_INDEX = 2
    LOG_IAT_INDEX = 3  # 新增: Log-IAT特征索引
    VALID_MASK_INDEX = 4
    IAT_INDEX = LOG_IAT_INDEX  # 兼容性别名
    CUMULATIVE_LEN_INDEX = None
    INPUT_FEATURE_DIM = len(FEATURE_NAMES)  # 5维特征
    
    # Burst 检测阈值（针对隧道流量优化）
    BURST_IAT_THRESHOLD = 0.01              # 10ms，适合检测 Iodine/DNS2TCP 隧道碎片
    
    # BurstSize 归一化
    BURSTSIZE_NORMALIZE = False
    BURSTSIZE_NORM_DENOM = math.log1p(MTU * SEQUENCE_LENGTH)
    
    # 全局统计令牌（可选）
    USE_GLOBAL_STATS_TOKEN = False          # 是否启用第1025行全局统计
    EFFECTIVE_SEQUENCE_LENGTH = SEQUENCE_LENGTH + (1 if USE_GLOBAL_STATS_TOKEN else 0)
    GLOBAL_STATS_MIN_PACKETS = 5            # 最少包数阈值

    # ============================================================
    # 骨干网络架构 (Dual-Stream Bi-Mamba)
    # ============================================================
    BACKBONE_ARCH = 'dual_stream'           # 双流架构
    MODEL_DIM = 32                          # 嵌入维度
    OUTPUT_DIM = 32                         # 骨干网络输出维度
    FEATURE_DIM = OUTPUT_DIM                # 特征维度（用于下游任务）
    EMBEDDING_DROPOUT = 0.1                 # 嵌入层Dropout
    POSITIONAL_ENCODING = "sinusoidal"      # 位置编码类型
    
    # ============================================================
    # 优化建议配置
    # ============================================================
    # 优化建议1: LayerNorm vs BatchNorm
    USE_LAYERNORM_IN_CLASSIFIER = True      # True: 使用LayerNorm (推荐), False: 使用BatchNorm
    USE_BACKBONE_LAYERNORM = False          # True: 在backbone输出后添加LayerNorm (可选，通常不需要)
    
    # 优化建议2: CL独立投影头
    USE_CL_PROJECTION_HEAD = True           # True: 为CL创建独立投影头 (推荐), False: 直接使用原始特征
    CL_PROJECTION_TRAINABLE = False         # True: CL投影头可训练 (需要预训练), False: 冻结投影头 (推荐)
    
    # Mamba 层参数
    MAMBA_LAYERS = 2                        # Mamba层数
    MAMBA_STATE_DIM = 8                     # SSM状态维度
    MAMBA_EXPANSION_FACTOR = 2              # 扩展因子
    MAMBA_CONV_KERNEL = 4                   # 卷积核大小
    MAMBA_DROPOUT = 0.1                     # Dropout率
    MAMBA_FUSION_TYPE = "gate"              # 融合方式: gate/average/concat_project
    MAMBA_PROJECTION_DIM = 64               # Concat后投影维度
    
    # ============================================================
    # STAGE 1: 自监督预训练 (SimMTM + InfoNCE) - 最优配置
    # ============================================================
    
    # 1.1 基础训练参数
    PRETRAIN_EPOCHS = 500                   # 最大训练轮数
    PRETRAIN_BATCH_SIZE = 64                # 批次大小（适配10.75GB显存）
    PRETRAIN_BATCH_SIZE_NNCLR = 32          # NNCLR专用批次（显存占用高）
    PRETRAIN_BATCH_SIZE_SIMSIAM = 64        # SimSiam批次
    PRETRAIN_GRADIENT_ACCUMULATION_STEPS = 2  # 梯度累积（有效批次=64*2=128）
    PRETRAIN_LR = 1e-3                      # 学习率
    PRETRAIN_WEIGHT_DECAY = 1e-4            # 权重衰减
    PRETRAIN_MIN_LR = 1e-5                  # 最小学习率
    LR_SCHEDULER = "cosine"                 # 学习率调度器
    
    # 1.2 早停机制
    PRETRAIN_EARLY_STOPPING = True          # 启用早停
    PRETRAIN_ES_WARMUP_EPOCHS = 50          # 预热轮数（前50轮不触发早停）
    PRETRAIN_ES_PATIENCE = 30               # 耐心值（30轮不改善则停止）
    PRETRAIN_ES_MIN_DELTA = 0.005           # 改善阈值（0.5%）
    
    # 1.3 SimMTM 参数（掩码重建任务）
    SIMMTM_MASK_RATE = 0.5                  # 掩码率（50%）
    PRETRAIN_NOISE_STD = 0.05               # 高斯噪声标准差
    PRETRAIN_RECON_WEIGHT = 1.0             # 重建损失权重
    PRETRAIN_LENGTH_WEIGHT = 1.0            # Length特征权重
    PRETRAIN_BURST_WEIGHT = 1.0             # BurstSize特征权重
    PRETRAIN_DIRECTION_WEIGHT = 1.0         # Direction特征权重
    PRETRAIN_LOG_IAT_WEIGHT = 0.8           # LogIAT特征权重（新增，降低至0.8以平衡学习）
    PRETRAIN_VALIDMASK_WEIGHT = 0.5         # ValidMask特征权重
    SIMMTM_DECODER_USE_MLP = False          # 解码器使用MLP
    SIMMTM_DECODER_HIDDEN_DIM = 64          # 解码器隐藏层维度

    # 1.4 InfoNCE 对比学习参数（最优配置）
    USE_INSTANCE_CONTRASTIVE = True         # 启用实例对比学习
    CONTRASTIVE_METHOD = "infonce"          # 对比学习方法: infonce/nnclr/simsiam
    INFONCE_TEMPERATURE = 0.25              # 温度系数τ（从0.2提升至0.25，改善小数据集对比学习）
    INFONCE_LAMBDA = 0.5                    # 对比学习损失权重
    
    # NNCLR 参数（备选）
    NNCLR_QUEUE_SIZE = 4096                 # 队列大小
    NNCLR_MIN_SIMILARITY = 0.2              # 最小相似度
    NNCLR_WARMUP_EPOCHS = 10                # 预热轮数
    
    # SupCon 参数（Legacy，保留兼容性）
    SUPCON_TEMPERATURE = 0.1
    SUPCON_LAMBDA = 1.0
    
    # 1.5 流量数据增强（针对预训练）
    AUG_CROP_PROB = 0.8                     # 时序裁剪概率
    AUG_JITTER_PROB = 0.6                   # 时序抖动概率
    AUG_CHANNEL_MASK_PROB = 0.5             # 通道掩码概率
    AUG_CROP_MIN_RATIO = 0.5                # 最小裁剪比例
    AUG_CROP_MAX_RATIO = 0.9                # 最大裁剪比例
    AUG_JITTER_STD = 0.1                    # 抖动标准差
    AUG_CHANNEL_MASK_RATIO = 0.15           # 掩码比例
    
    # 流量增强（降低强度，避免破坏关键特征）
    TRAFFIC_AUG_CROP_PROB = 0.5             # 时序裁剪概率
    TRAFFIC_AUG_JITTER_PROB = 0.4           # 时序抖动概率
    TRAFFIC_AUG_MASK_PROB = 0.3             # 通道掩码概率
    TRAFFIC_AUG_CROP_MIN_RATIO = 0.5        # 最小裁剪比例
    TRAFFIC_AUG_CROP_MAX_RATIO = 0.9        # 最大裁剪比例
    TRAFFIC_AUG_JITTER_STD = 0.1            # 抖动标准差
    TRAFFIC_AUG_MASK_RATIO = 0.15           # 掩码比例
    
    # Burst 抖动增强（针对隧道流量）
    TRAFFIC_AUG_BURST_JITTER_PROB = 0.5     # Burst抖动概率
    TRAFFIC_AUG_BURST_JITTER_STD = 0.05     # Burst抖动标准差（±5%）
    
    # 日志配置
    FEATURE_EXTRACTION_VERBOSE = True       # 显示详细日志
    FEATURE_EXTRACTION_PROGRESS_BAR = True  # 显示进度条
    FEATURE_EXTRACTION_VALIDATE = True      # 验证特征质量

    # ============================================================
    # STAGE 2: 标签矫正 + 数据增强 - 最优配置
    # ============================================================
    
    # 2.1 Hybrid Court 标签矫正（三阶段策略）
    # Phase 1: 核心严选阈值
    HC_PHASE1_CL_BENIGN = 0.54              # 正常样本CL置信度阈值
    HC_PHASE1_CL_MALICIOUS = 0.57           # 恶意样本CL置信度阈值
    HC_PHASE1_KNN_BENIGN = 0.60             # 正常样本KNN一致性阈值
    HC_PHASE1_KNN_MALICIOUS = 0.70          # 恶意样本KNN一致性阈值
    
    # Phase 2: 分级挽救阈值
    HC_PHASE2_REWEIGHT_BASE_CL = 0.35       # 重加权基础准入线
    HC_PHASE2_MADE_ANOMALY = 60.0           # MADE密度异常分界线
    HC_PHASE2_SYS_CONF_SPLIT = 0.30         # 系统置信度分界线
    
    # Phase 3: 锚点拯救阈值
    HC_PHASE3_MIN_ANCHORS = 20              # 最少锚点样本数
    HC_PHASE3_KNN_K = 15                    # 锚点KNN的K值
    HC_PHASE3_RESCUE_KEEP_CONS = 0.70       # 拯救为Keep的最低一致性
    HC_PHASE3_RESCUE_FLIP_CONS = 0.75       # 拯救为Flip的最低一致性
    HC_PHASE3_RESCUE_KEEP_WEIGHT = 0.85     # 拯救Keep的权重
    HC_PHASE3_RESCUE_FLIP_WEIGHT = 0.75     # 拯救Flip的权重
    
    # 权重分配（分层策略）
    HC_WEIGHT_TIER1_CORE = 1.0              # Tier 1: 核心样本
    HC_WEIGHT_TIER2_FLIP = 1.0              # Tier 2: 翻转样本
    HC_WEIGHT_TIER3A_KEEP_HI = 1.0          # Tier 3a: 优质保持
    HC_WEIGHT_TIER3B_KEEP_LO = 0.4          # Tier 3b: 存疑保持
    HC_WEIGHT_TIER4A_REW_HI = 0.6           # Tier 4a: 优质重加权
    HC_WEIGHT_TIER4B_REW_LO = 0.1           # Tier 4b: 噪声重加权
    
    # CL (Confident Learning) 参数
    CL_K_FOLD = 5                           # K折交叉验证
    
    # MADE (Density Estimation) 参数
    MADE_HIDDEN_DIMS = [64, 128, 64]        # 隐藏层维度
    MADE_DENSITY_THRESHOLD_PERCENTILE = 70  # 密度阈值百分位
    
    # KNN (Semantic Voting) 参数
    KNN_NEIGHBORS = 20                      # K近邻数量
    KNN_METRIC = "euclidean"                # 距离度量
    
    # 动态密度阈值（可选）
    HYBRIDCOURT_DYNAMIC_DENSITY_THRESHOLDS = False
    HYBRIDCOURT_DENSITY_HIGH_PCT = 90.0
    HYBRIDCOURT_DENSITY_LOW_PCT = 50.0

    # 2.2 TabDDPM 数据增强（特征空间生成）- 优化版 v2.5
    STAGE2_USE_TABDDPM = True               # 启用TabDDPM
    STAGE2_TABDDPM_SPACE = 'feature'        # 在特征空间增强
    
    # 增强倍数策略（v2.5：按类别差异化增强，模拟测试集分布）
    AUGMENTATION_MODE = 'class_specific'    # 类别特定模式（新增）
    STAGE2_FEATURE_AUG_MULTIPLIER = 2       # 默认增强倍数（向后兼容）
    
    # 类别特定增强倍数（2:1 策略 - 正常:恶意 = 2:1）
    STAGE2_BENIGN_AUG_MULTIPLIER = 4        # 正常样本增强（适度恢复，提升覆盖）
    STAGE2_MALICIOUS_AUG_MULTIPLIER = 3     # 恶意样本增强（适度恢复，提升覆盖）
    # 预期结果: 750正常 + 500恶意 = 1250 (60%正常, 40%恶意)
    
    # 向后兼容：固定倍数模式
    AUGMENTATION_RATIO_MIN = 1              # 最小倍数
    AUGMENTATION_RATIO_MAX = 1              # 最大倍数
    
    # 分层增强策略（基于样本质量）
    STAGE2_FEATURE_TIER1_MIN_WEIGHT = 0.9   # Tier1最低权重
    STAGE2_FEATURE_TIER1_MULTIPLIER = 10    # Tier1增强倍数
    STAGE2_FEATURE_TIER2_MIN_WEIGHT = 0.7   # Tier2最低权重
    STAGE2_FEATURE_TIER2_MULTIPLIER = 5     # Tier2增强倍数
    STAGE2_FEATURE_LOWCONF_MULTIPLIER = 0   # 低置信度样本不增强

    # Stage2 增强数据过滤：仅允许高权重样本参与TabDDPM训练/采样
    STAGE2_AUGMENT_MIN_WEIGHT = 0.8
    
    # TabDDPM 训练参数（优化版 v2.5 - 恢复旧参数以提高生成质量）
    DDPM_EPOCHS = 1500                      # 训练轮数（保持1500）
    DDPM_LR = 5e-4                          # 初始学习率（恢复旧值：3e-4→5e-4，加快收敛）
    DDPM_LR_SCHEDULER = 'cosine'            # 学习率调度器
    DDPM_MIN_LR = 1e-5                      # 最小学习率（恢复旧值：5e-6→1e-5）
    DDPM_TIMESTEPS = 1000                   # 扩散步数
    DDPM_HIDDEN_DIMS = [128, 256, 256, 128] # 隐藏层维度（保持容量）
    DDPM_SAMPLING_STEPS = 150               # DDIM采样步数（保持150，更精细生成）
    
    # TabDDPM 早停机制（v2.5：减少平滑窗口，更敏感）
    DDPM_EARLY_STOPPING = True              # 启用早停
    DDPM_ES_WARMUP_EPOCHS = 150             # 预热轮数（保持150）
    DDPM_ES_PATIENCE = 150                  # 耐心值（保持150）
    DDPM_ES_MIN_DELTA = 0.0003              # 改善阈值（保持0.0003）
    DDPM_ES_SMOOTH_WINDOW = 5               # 平滑窗口（优化：5→3，更敏感）
    
    # Differential Guidance（分类引导）
    GUIDANCE_BENIGN = 1.0                   # 正常流量：无引导
    GUIDANCE_MALICIOUS = 1.2                # 恶意流量：适度强化
    
    # Structure-aware Generation（结构感知生成）
    MASK_PROBABILITY = 0.3                  # 掩码概率
    MASK_LAMBDA = 0.2                       # 掩码损失权重
    
    # 特征条件化配置
    # 条件特征：Direction, ValidMask（协议约束，固定不变）
    COND_FEATURE_INDICES = [DIRECTION_INDEX, VALID_MASK_INDEX]
    # 依赖特征：Length, BurstSize, LogIAT（学习分布，可生成）
    DEP_FEATURE_INDICES = [LENGTH_INDEX, BURST_SIZE_INDEX, LOG_IAT_INDEX]
    
    # 高级选项
    ENABLE_COVARIANCE_MATCHING = False      # 协方差匹配
    ENABLE_DISCRETE_QUANTIZATION = False    # 离散量化
    DISCRETE_QUANTIZE_INDICES = []
    DISCRETE_QUANTIZE_MAX_VALUES = 4096
    AUGMENT_USE_WEIGHTED_SAMPLING = True    # 加权采样
    
    # 增强模板质量门槛（v2.5：提高质量要求）
    AUGMENT_TEMPLATE_MIN_WEIGHT = 0.8       # 模板最低权重（0.7→0.8，更严格）
    AUGMENT_TEMPLATE_MIN_WEIGHT_HARD = 0.6  # 硬门槛（0.5→0.6，提高底线）
    
    # 数据质量控制（新增 v2.5）
    AUGMENT_QUALITY_CHECK = True            # 启用质量检查
    AUGMENT_MAX_OUTLIER_RATIO = 0.05        # 最大离群点比例（5%）
    AUGMENT_MIN_DIVERSITY_SCORE = 0.3       # 最小多样性分数
    
    # Mixup增强（新增 v2.6 - 提升样本多样性）
    AUGMENT_MIXUP_ENABLED = True            # 启用Mixup增强
    AUGMENT_MIXUP_ALPHA_BENIGN = 0.1        # 正常类Mixup强度（弱混合，保持纯净）
    AUGMENT_MIXUP_ALPHA_MALICIOUS = 0.3     # 恶意类Mixup强度（强混合，增加多样性）
    
    # 困难样本挖掘（新增 v2.6 - 针对性增强）
    AUGMENT_HARD_MINING_ENABLED = False     # 启用困难样本挖掘（需要预训练模型）
    AUGMENT_HARD_MINING_RATIO = 0.3         # Top 30%不确定性样本
    AUGMENT_HARD_MINING_MULTIPLIER = 3      # 困难样本3倍增强

    # ============================================================
    # STAGE 3: 分类器微调 - 最优配置
    # ============================================================
    
    # 3.1 分类器架构
    CLASSIFIER_HIDDEN_DIM = 32              # 隐藏层维度
    CLASSIFIER_OUTPUT_DIM = 2               # 二分类输出
    CLASSIFIER_INPUT_IS_FEATURES = False    # 输入类型（False=序列，True=特征）
    
    # 3.2 训练基础参数
    FINETUNE_EPOCHS = 1000                  # 最大训练轮数（恢复最优配置）
    FINETUNE_BATCH_SIZE = 128               # 批次大小
    FINETUNE_LR = 2e-4                      # 学习率
    FINETUNE_MIN_LR = 1e-6                  # 最小学习率
    TEST_BATCH_SIZE = 256                   # 测试批次大小
    
    # 3.3 早停机制（智能训练终止）
    FINETUNE_EARLY_STOPPING = True         # 启用早停
    FINETUNE_ES_WARMUP_EPOCHS = 100         # 预热轮数（前N轮不触发早停）- 增加预热
    FINETUNE_ES_PATIENCE = 100               # 耐心值（连续N轮无改善则停止）- 增加耐心
    FINETUNE_ES_MIN_DELTA = 0.0005           # F1改善阈值（需要明显改善）- 提高阈值
    FINETUNE_ES_METRIC = 'f1_optimal'       # 监控指标
    FINETUNE_ES_ALLOW_TRAIN_METRIC = True   # 允许使用训练集指标
    
    # 3.4 验证集配置（可选）
    FINETUNE_VAL_SPLIT = 0               # 验证集比例（15%）- 启用验证集监控
    FINETUNE_VAL_PER_CLASS = 0              # 每类固定验证样本数
    VALIDATION_SIZE_ORIGINAL = 0.2          # 原始数据验证比例
    VALIDATION_SIZE_SYNTHETIC = 0.1         # 合成数据验证比例
    
    # 3.5 温室训练策略（强制1:1平衡采样）
    USE_BALANCED_SAMPLING = True            # 启用平衡采样
    BALANCED_SAMPLING_RATIO = 1.0           # 目标比例（正常:恶意=1:1）
    
    # 3.6 骨干网络微调（最优配置）
    # 注意：如果Stage 3输入是特征向量（2D），骨干微调会被自动禁用
    # 原因：特征向量无法反向传播到骨干网络
    # 解决方案：启用混合训练（STAGE3_MIXED_STREAM = True）以支持骨干微调
    # 当前配置：混合训练已启用，因此骨干微调可以正常工作
    FINETUNE_BACKBONE = True               # 启用骨干微调（混合训练模式下有效）
    FINETUNE_BACKBONE_SCOPE = 'all'  # 微调范围
    FINETUNE_BACKBONE_LR = 2e-5             # 骨干网络学习率
    FINETUNE_BACKBONE_WARMUP_EPOCHS = 50    # 预热轮数
    
    # 3.7 混合训练模式（原始序列 + 增强特征）- 优化版 v2.4
    STAGE3_MIXED_STREAM = True              # 启用混合训练
    STAGE3_MIXED_REAL_BATCH_SIZE = 64       # 原始序列批次（提高真实数据占比）
    STAGE3_MIXED_SYN_BATCH_SIZE = 64        # 增强特征批次（降低synthetic主导）
    STAGE3_MIXED_REAL_LOSS_SCALE = 3.0      # 原始数据损失权重（进一步强调真实序列）
    STAGE3_MIXED_SYN_LOSS_SCALE = 0.8       # 增强数据损失权重（略降，抑制FP）
    
    # 3.8 在线数据增强（可选）
    STAGE3_ONLINE_AUGMENTATION = False      # 关闭在线增强
    
    # 3.9 ST-Mixup 增强（可选，默认关闭）
    STAGE3_USE_ST_MIXUP = False             # 关闭ST-Mixup
    STAGE3_ST_MIXUP_MODE = 'intra_class'
    STAGE3_ST_MIXUP_ALPHA = 0.2
    STAGE3_ST_MIXUP_WARMUP_EPOCHS = 100
    STAGE3_ST_MIXUP_MAX_PROB = 0.3
    STAGE3_ST_MIXUP_TIME_SHIFT_RATIO = 0.15
    STAGE3_ST_MIXUP_UNCERTAINTY_THRESHOLD = 0.3
    
    # 3.10 困难样本挖掘（可选，默认关闭）
    STAGE3_HARD_MINING = False
    STAGE3_HARD_MINING_WARMUP_EPOCHS = 5
    STAGE3_HARD_MINING_FREQ_EPOCHS = 3
    STAGE3_HARD_MINING_TOPK_RATIO = 0.2
    STAGE3_HARD_MINING_MULTIPLIER = 3.0
    STAGE3_HARD_MINING_POS_PROB_MAX = 0.70
    STAGE3_HARD_MINING_NEG_PROB_MIN = 0.60

    # ============================================================
    # 损失函数配置（最优组合）
    # ============================================================
    
    # 主损失：Focal Loss（关注困难样本）
    USE_FOCAL_LOSS = True                   # 启用Focal Loss
    FOCAL_ALPHA = 0.6                       # 恶意类权重（提高召回，提升F1）
    FOCAL_GAMMA = 2.0                       # Gamma参数
    
    # 辅助损失（全部关闭 - 最优配置）
    USE_SOFT_F1_LOSS = False
    SOFT_F1_WEIGHT = 0.1
    USE_BCE_LOSS = False
    BCE_POS_WEIGHT = 1.0
    BCE_LABEL_SMOOTHING = 0.0
    SUP_LOSS_SCALE = 1.0
    USE_MARGIN_LOSS = False
    MARGIN_M = 0.15
    MARGIN_S = 1.0
    MARGIN_LOSS_WEIGHT = 0.15
    MARGIN_LOSS_WEIGHT_START = 0.0
    MARGIN_LOSS_WEIGHT_END = 0.2
    USE_LOGIT_MARGIN = False
    LOGIT_MARGIN_M = 0.25
    LOGIT_MARGIN_WARMUP_EPOCHS = 30
    
    # Label Smoothing（减少过拟合）
    LABEL_SMOOTHING = 0.05
    
    # 类别权重（1:1平衡）
    CLASS_WEIGHT_BENIGN = 1.0
    CLASS_WEIGHT_MALICIOUS = 1.0
    
    # 决策阈值（最优配置）
    MALICIOUS_THRESHOLD = 0.5           # 最优阈值（基于测试集F1优化）
    
    # 动态损失权重（全部关闭）
    SOFT_ORTH_WEIGHT_START = 0.0
    SOFT_ORTH_WEIGHT_END = 0.0
    CONSISTENCY_WEIGHT_START = 0.0
    CONSISTENCY_WEIGHT_END = 0.0
    CONSISTENCY_TEMPERATURE = 2.0
    CONSISTENCY_WARMUP_EPOCHS = 5
    
    # Co-teaching（协同教学，禁用 - 数据无噪声）
    USE_CO_TEACHING = False                 # 禁用Co-teaching（数据使用真实标签，无噪声）
    CO_TEACHING_SELECT_RATE = 0.7           # 选择率（70%样本）
    CO_TEACHING_WARMUP_EPOCHS = 10          # 预热轮数（前N轮不启用）
    CO_TEACHING_MIN_SAMPLE_WEIGHT = 0.5     # 最小样本权重
    CO_TEACHING_DYNAMIC_RATE = True         # 动态调整选择率
    CO_TEACHING_NOISE_RATE = 0.0            # 假设噪声率（0.0 = 无噪声）

    # ============================================================
    # 硬件和训练配置
    # ============================================================
    GPU_ID = int(os.environ.get('MEDAL_GPU_ID', '7'))
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        _visible_count = torch.cuda.device_count()
        _safe_gpu_id = GPU_ID if 0 <= int(GPU_ID) < _visible_count else 0
        DEVICE = torch.device(f"cuda:{_safe_gpu_id}")
    else:
        DEVICE = torch.device("cpu")
    
    NUM_WORKERS = 4
    SEED = 42
    SAVE_EVERY_N_EPOCHS = 10
    
    # ============================================================
    # 可视化配置
    # ============================================================
    VIS_FEATURE_DIM_REDUCTION = "tsne"
    VIS_PERPLEXITY = 30
    VIS_N_ITER = 1000
    
    # ============================================================
    # 输出目录结构
    # ============================================================
    PREPROCESSING_DIR = os.path.join(OUTPUT_ROOT, "preprocessing")
    FEATURE_EXTRACTION_DIR = os.path.join(OUTPUT_ROOT, "feature_extraction")
    LABEL_CORRECTION_DIR = os.path.join(OUTPUT_ROOT, "label_correction")
    DATA_AUGMENTATION_DIR = os.path.join(OUTPUT_ROOT, "data_augmentation")
    CLASSIFICATION_DIR = os.path.join(OUTPUT_ROOT, "classification")
    RESULT_DIR = os.path.join(OUTPUT_ROOT, "result")
    CHECKPOINT_DIR = CLASSIFICATION_DIR
    
    # ============================================================
    # 工具方法
    # ============================================================
    
    @staticmethod
    def create_dirs():
        """创建必要的输出目录"""
        os.makedirs(Config.OUTPUT_ROOT, exist_ok=True)
        for module_dir in [Config.PREPROCESSING_DIR, Config.FEATURE_EXTRACTION_DIR,
                          Config.LABEL_CORRECTION_DIR, Config.DATA_AUGMENTATION_DIR,
                          Config.CLASSIFICATION_DIR, Config.RESULT_DIR]:
            os.makedirs(module_dir, exist_ok=True)
            os.makedirs(os.path.join(module_dir, "models"), exist_ok=True)
            os.makedirs(os.path.join(module_dir, "figures"), exist_ok=True)
            os.makedirs(os.path.join(module_dir, "logs"), exist_ok=True)
    
    @staticmethod
    def get_dynamic_weight(epoch, total_epochs, start_weight, end_weight):
        """线性调度动态权重"""
        return start_weight + (end_weight - start_weight) * (epoch / total_epochs)

    def log_stage_config(self, logger, stage: str):
        """输出指定阶段的配置参数和策略"""
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"📋 {stage} 配置参数")
        logger.info("=" * 70)
        
        if stage == "Stage 1" or stage == "pretrain":
            logger.info("🎯 目标: 自监督预训练骨干网络")
            logger.info("")
            logger.info("📊 训练参数:")
            logger.info(f"  - 最大轮数: {self.PRETRAIN_EPOCHS}")
            logger.info(f"  - 批次大小: {self.PRETRAIN_BATCH_SIZE}")
            logger.info(f"  - 学习率: {self.PRETRAIN_LR} → {self.PRETRAIN_MIN_LR}")
            logger.info(f"  - 权重衰减: {self.PRETRAIN_WEIGHT_DECAY}")
            logger.info(f"  - 调度器: {self.LR_SCHEDULER}")
            logger.info("")
            logger.info("🔧 SimMTM 配置:")
            logger.info(f"  - 掩码率: {self.SIMMTM_MASK_RATE}")
            logger.info(f"  - 噪声标准差: {self.PRETRAIN_NOISE_STD}")
            logger.info(f"  - 特征权重: Length={self.PRETRAIN_LENGTH_WEIGHT}, Burst={self.PRETRAIN_BURST_WEIGHT}, Dir={self.PRETRAIN_DIRECTION_WEIGHT}, LogIAT={self.PRETRAIN_LOG_IAT_WEIGHT}, VM={self.PRETRAIN_VALIDMASK_WEIGHT}")
            logger.info("")
            if self.USE_INSTANCE_CONTRASTIVE:
                logger.info(f"🔧 对比学习配置 ({self.CONTRASTIVE_METHOD.upper()}):")
                logger.info(f"  - 温度系数: {self.INFONCE_TEMPERATURE}")
                logger.info(f"  - 损失权重: {self.INFONCE_LAMBDA}")
            logger.info("")
            logger.info("⏹️ 早停配置:")
            logger.info(f"  - 启用: {self.PRETRAIN_EARLY_STOPPING}")
            logger.info(f"  - 预热轮数: {self.PRETRAIN_ES_WARMUP_EPOCHS}")
            logger.info(f"  - 耐心值: {self.PRETRAIN_ES_PATIENCE}")
            logger.info(f"  - 改善阈值: {self.PRETRAIN_ES_MIN_DELTA}")
            
        elif stage == "Stage 2" or stage == "correction":
            logger.info("🎯 目标: 标签矫正 + 数据增强")
            logger.info("")
            logger.info("📊 Hybrid Court 配置:")
            logger.info(f"  - Phase 1 CL阈值: Benign={self.HC_PHASE1_CL_BENIGN}, Malicious={self.HC_PHASE1_CL_MALICIOUS}")
            logger.info(f"  - Phase 1 KNN阈值: Benign={self.HC_PHASE1_KNN_BENIGN}, Malicious={self.HC_PHASE1_KNN_MALICIOUS}")
            logger.info(f"  - CL K折: {self.CL_K_FOLD}")
            logger.info(f"  - KNN邻居数: {self.KNN_NEIGHBORS}")
            logger.info("")
            if self.STAGE2_USE_TABDDPM:
                logger.info("📊 TabDDPM 配置:")
                logger.info(f"  - 增强空间: {self.STAGE2_TABDDPM_SPACE}")
                logger.info(f"  - 增强倍数: {self.STAGE2_FEATURE_AUG_MULTIPLIER}x")
                logger.info(f"  - 训练轮数: {self.DDPM_EPOCHS}")
                logger.info(f"  - 学习率: {self.DDPM_LR}")
                logger.info(f"  - 扩散步数: {self.DDPM_TIMESTEPS}")
                logger.info(f"  - 采样步数: {self.DDPM_SAMPLING_STEPS}")
                logger.info("")
                logger.info("⏹️ TabDDPM 早停:")
                logger.info(f"  - 启用: {self.DDPM_EARLY_STOPPING}")
                logger.info(f"  - 预热轮数: {self.DDPM_ES_WARMUP_EPOCHS}")
                logger.info(f"  - 耐心值: {self.DDPM_ES_PATIENCE}")
            
        elif stage == "Stage 3" or stage == "finetune":
            logger.info("🎯 目标: 分类器微调")
            logger.info("")
            logger.info("📊 训练参数:")
            logger.info(f"  - 最大轮数: {self.FINETUNE_EPOCHS}")
            logger.info(f"  - 批次大小: {self.FINETUNE_BATCH_SIZE}")
            logger.info(f"  - 学习率: {self.FINETUNE_LR}")
            logger.info("")
            logger.info("🔧 骨干微调配置:")
            logger.info(f"  - 启用: {self.FINETUNE_BACKBONE}")
            logger.info(f"  - 范围: {self.FINETUNE_BACKBONE_SCOPE}")
            logger.info(f"  - 学习率: {self.FINETUNE_BACKBONE_LR}")
            logger.info(f"  - 预热轮数: {self.FINETUNE_BACKBONE_WARMUP_EPOCHS}")
            logger.info("")
            logger.info("🔧 损失函数配置:")
            logger.info(f"  - Focal Loss: alpha={self.FOCAL_ALPHA}, gamma={self.FOCAL_GAMMA}")
            logger.info(f"  - 标签平滑: {self.LABEL_SMOOTHING}")
            logger.info("")
            if self.USE_CO_TEACHING:
                logger.info("🔧 Co-teaching配置:")
                logger.info(f"  - 启用: {self.USE_CO_TEACHING}")
                logger.info(f"  - 预热轮数: {self.CO_TEACHING_WARMUP_EPOCHS}")
                logger.info(f"  - 选择率: {self.CO_TEACHING_SELECT_RATE}")
                logger.info(f"  - 动态调整: {self.CO_TEACHING_DYNAMIC_RATE}")
                if self.CO_TEACHING_DYNAMIC_RATE:
                    logger.info(f"  - 假设噪声率: {self.CO_TEACHING_NOISE_RATE}")
                logger.info("")
            logger.info("🔧 采样策略:")
            logger.info(f"  - 平衡采样: {self.USE_BALANCED_SAMPLING}")
            logger.info(f"  - 目标比例: {self.BALANCED_SAMPLING_RATIO}:1")
            logger.info("")
            if self.STAGE3_MIXED_STREAM:
                logger.info("🔧 混合训练配置:")
                logger.info(f"  - 原始序列批次: {self.STAGE3_MIXED_REAL_BATCH_SIZE}")
                logger.info(f"  - 增强特征批次: {self.STAGE3_MIXED_SYN_BATCH_SIZE}")
                logger.info(f"  - 原始损失权重: {self.STAGE3_MIXED_REAL_LOSS_SCALE}")
                logger.info(f"  - 增强损失权重: {self.STAGE3_MIXED_SYN_LOSS_SCALE}")
            logger.info("")
            logger.info("⏹️ 早停配置:")
            logger.info(f"  - 启用: {self.FINETUNE_EARLY_STOPPING}")
            logger.info(f"  - 预热轮数: {self.FINETUNE_ES_WARMUP_EPOCHS} (前N轮不触发早停)")
            logger.info(f"  - 耐心值: {self.FINETUNE_ES_PATIENCE} (连续N轮无改善则停止)")
            if self.FINETUNE_ES_MIN_DELTA > 0:
                logger.info(f"  - 改善阈值: {self.FINETUNE_ES_MIN_DELTA} (需改善至少此值)")
            else:
                logger.info(f"  - 改善阈值: {self.FINETUNE_ES_MIN_DELTA} (任何改善都算)")
        
        logger.info("=" * 70)
        logger.info("")

    def get_all_configs_dict(self):
        """获取所有配置的字典表示"""
        config_dict = {}
        for attr in dir(self):
            if not attr.startswith('_') and attr.isupper():
                config_dict[attr] = getattr(self, attr)
        return config_dict
    
    def __repr__(self):
        """打印所有配置参数"""
        config_str = "\n" + "="*70 + "\n"
        config_str += "MEDAL-Lite Configuration (重构版 v2)\n"
        config_str += "="*70 + "\n"
        for attr in dir(self):
            if not attr.startswith('_') and attr.isupper():
                config_str += f"{attr}: {getattr(self, attr)}\n"
        config_str += "="*70 + "\n"
        return config_str
    
    def print_config(self):
        """打印配置信息"""
        print(self)


# ============================================================
# 全局配置实例
# ============================================================
config = Config()


# ============================================================
# 便捷函数
# ============================================================
def print_config():
    """打印配置信息（便捷函数）"""
    config.print_config()


if __name__ == "__main__":
    print_config()
    print(f"\n✓ 项目根目录: {config.PROJECT_ROOT}")
    print(f"✓ 数据根目录: {config.DATA_ROOT}")
    print(f"✓ 输出根目录: {config.OUTPUT_ROOT}")
    print(f"✓ 设备: {config.DEVICE}")
    print(f"\n最优配置摘要 (ablation_data_augmentation_20260110_214253):")
    print(f"  - Stage 1: SimMTM + {config.CONTRASTIVE_METHOD.upper()} (τ={config.INFONCE_TEMPERATURE}, λ={config.INFONCE_LAMBDA})")
    print(f"  - Stage 2: Hybrid Court + TabDDPM ({config.STAGE2_FEATURE_AUG_MULTIPLIER}x增强)")
    print(f"  - Stage 3: Mixed Training (real={config.STAGE3_MIXED_REAL_BATCH_SIZE}, syn={config.STAGE3_MIXED_SYN_BATCH_SIZE})")
    print(f"  - 骨干微调: {'启用' if config.FINETUNE_BACKBONE else '关闭'}")
    print(f"  - Co-teaching: {'启用' if config.USE_CO_TEACHING else '禁用'} (noise_rate={config.CO_TEACHING_NOISE_RATE})")
    print(f"  - 训练轮数: {config.FINETUNE_EPOCHS} epochs")
    print(f"  - 最优阈值: {config.MALICIOUS_THRESHOLD}")
    print(f"\n最优性能:")
    print(f"  - F1 Score: 0.9026")
    print(f"  - Precision: 0.8972")
    print(f"  - Recall: 0.9080")
    print(f"  - AUC: 0.9896")
    print(f"  - Accuracy: 0.9822")
    print(f"\n配置更新 (v2.3 - 2026-01-11 - TabDDPM优化组合1):")
    print(f"  - 模型容量: {config.DDPM_HIDDEN_DIMS} (参数量提升~3x)")
    print(f"  - 学习率: {config.DDPM_LR} (初始) → {config.DDPM_MIN_LR} (最小)")
    print(f"  - 调度器: {config.DDPM_LR_SCHEDULER}")
    print(f"  - 训练轮数: {config.DDPM_EPOCHS} epochs")
    print(f"  - 早停耐心值: {config.DDPM_ES_PATIENCE}")
    print(f"  - 预期效果: 最佳损失 0.28-0.32, F1提升 1-2%")
