"""
Configuration file for MEDAL-Lite Model
Contains all hyperparameters and settings

统一配置文件 - 合并自根目录config.py和MoudleCode/utils/config.py
所有模块统一使用此配置: from MoudleCode.utils.config import config
"""
import torch
import os
from pathlib import Path

class Config:
    """Global configuration for MEDAL-Lite"""
    
    # ==================== Paths ====================
    # 使用动态路径，自动定位项目根目录
    PROJECT_ROOT = str(Path(__file__).parent.parent.parent.absolute())
    DATASET_SUBDIR = os.environ.get('MEDAL_DATASET_SUBDIR', '').strip()
    if DATASET_SUBDIR:
        DATA_ROOT = os.path.join(PROJECT_ROOT, "Datasets", DATASET_SUBDIR)
    else:
        DATA_ROOT = os.path.join(PROJECT_ROOT, "Datasets")

    DATASET_NAME = os.environ.get('MEDAL_DATASET_NAME', '').strip()
    if DATASET_NAME:
        OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", DATASET_NAME)
    else:
        OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")
    
    # Fixed paths - read all pcap files from these directories
    BENIGN_TRAIN = os.path.join(DATA_ROOT, "T1_train", "benign")
    MALICIOUS_TRAIN = os.path.join(DATA_ROOT, "T1_train", "malicious")
    BENIGN_TEST = os.path.join(DATA_ROOT, "T2_test", "benign")
    MALICIOUS_TEST = os.path.join(DATA_ROOT, "T2_test", "malicious")
    
    # ==================== Dataset Settings ====================
    # Flow counts will be counted during processing, not pre-specified
    LABEL_NOISE_RATE = 0.30  # 30% label noise
    
    # Labels
    LABEL_BENIGN = 0
    LABEL_MALICIOUS = 1
    
    # ==================== Preprocessing Parameters ====================
    # PCAP解析参数
    PACKET_TIMEOUT = 60  # 流超时时间(秒)
    MAX_PACKETS_PER_FLOW = 1024  # 每个流最大包数

    # -------------------- Fixed Feature Interface (New Design) --------------------
    # The project is standardized to a 4D feature set:
    # [Length, Direction, BurstSize, ValidMask]
    FEATURE_SET = 'lite4'

    FEATURE_NAMES = ['Length', 'Direction', 'BurstSize', 'ValidMask']
    LENGTH_INDEX = 0
    IAT_INDEX = None
    DIRECTION_INDEX = 1
    BURST_SIZE_INDEX = 2
    CUMULATIVE_LEN_INDEX = None
    VALID_MASK_INDEX = 3

    # Burst 检测阈值（秒）
    # 用于判定突发边界：当包间隔 > 阈值时，认为是新的突发
    # 
    # 推荐值：
    # - 0.1s (默认): 通用场景，适合大多数流量
    # - 0.05s: 中等敏感度，能捕捉更细粒度的突发
    # - 0.01s: 高敏感度，适合检测隧道流量（如 Iodine）的碎片化传输
    # 
    # 针对 DoHBrw 数据集（包含 Iodine/DNS2TCP 隧道）：
    # - Iodine 会将数据切成大量小包快速发送（微秒级间隔）
    # - 使用 0.01s 可以更好地捕捉这种"密集碎片"模式
    # - 避免将碎片合并成一个大 Burst，丢失隧道特征
    BURST_IAT_THRESHOLD = 0.01  # 从 0.1 降低到 0.01，更好地检测隧道碎片
    
    # 特征归一化参数
    MTU = 1500.0  # 最大传输单元
    TCP_WINDOW_MAX = 65535.0  # TCP窗口最大值
    IAT_EPSILON = 1e-7  # Log-IAT 防止log(0)
    
    # ==================== Input & Embedding Parameters ====================
    SEQUENCE_LENGTH = 1024  # L: Maximum number of packets per flow
    INPUT_FEATURE_DIM = 4
    MODEL_DIM = 32          # d_model: Embedding dimension (降低到32维)
    OUTPUT_DIM = 32         # backbone最终输出维度（下游分类/对比学习使用）
    FEATURE_DIM = OUTPUT_DIM
    EMBEDDING_DROPOUT = 0.1
    POSITIONAL_ENCODING = "sinusoidal"  # 正弦位置编码
    
    # ==================== Micro-Bi-Mamba Backbone ====================
    BACKBONE_ARCH = 'dual_stream'

    MAMBA_LAYERS = 2
    MAMBA_STATE_DIM = 8         # d_state: SSM internal memory capacity (降低到8)
    MAMBA_EXPANSION_FACTOR = 2   # E: Expansion factor
    MAMBA_CONV_KERNEL = 4        # Local conv kernel size
    MAMBA_DROPOUT = 0.1
    MAMBA_FUSION_TYPE = "concat_project"  # "average" 或 "concat_project"
    MAMBA_PROJECTION_DIM = 64    # Concat后的投影维度(32*2)
    
    # ==================== Pre-training (Stage 1) ====================
    PRETRAIN_EPOCHS = 500  # 给 Stage 1 充分收敛时间（默认上限）
    PRETRAIN_BATCH_SIZE = 64  # 降低批次大小以适应10.75GB显存 (从128降至64)
    PRETRAIN_BATCH_SIZE_NNCLR = 64  # NNCLR 专用批次大小（显存占用高，需要更小）
    PRETRAIN_GRADIENT_ACCUMULATION_STEPS = 2  # 梯度累积步数，有效批次 = 64 * 2 = 128
                                              # 仅用于 NNCLR，保持与其他方法相同的有效批次
    PRETRAIN_LR = 1e-3
    PRETRAIN_WEIGHT_DECAY = 1e-4
    PRETRAIN_MIN_LR = 1e-5
    PRETRAIN_EARLY_STOPPING = True  # 启用早停机制
    PRETRAIN_ES_WARMUP_EPOCHS = 50
    PRETRAIN_ES_PATIENCE = 30  # 从 20 增加到 30，更宽容的早停策略
    PRETRAIN_ES_MIN_DELTA = 0.005  # 从 0.01 降低到 0.005，更敏感的改进检测
    
    # SimMTM parameters
    SIMMTM_MASK_RATE = float(os.environ.get('MEDAL_SIMMTM_MASK_RATE', 0.5))  # 50% masking

    AUG_CROP_PROB = float(os.environ.get('MEDAL_AUG_CROP_PROB', 0.8))
    AUG_JITTER_PROB = float(os.environ.get('MEDAL_AUG_JITTER_PROB', 0.6))
    AUG_CHANNEL_MASK_PROB = float(os.environ.get('MEDAL_AUG_CHANNEL_MASK_PROB', 0.5))
    AUG_CROP_MIN_RATIO = float(os.environ.get('MEDAL_AUG_CROP_MIN_RATIO', 0.5))
    AUG_CROP_MAX_RATIO = float(os.environ.get('MEDAL_AUG_CROP_MAX_RATIO', 0.9))
    AUG_JITTER_STD = float(os.environ.get('MEDAL_AUG_JITTER_STD', 0.1))
    AUG_CHANNEL_MASK_RATIO = float(os.environ.get('MEDAL_AUG_CHANNEL_MASK_RATIO', 0.15))

    SIMMTM_DECODER_USE_MLP = bool(int(os.environ.get('MEDAL_SIMMTM_DECODER_USE_MLP', 0)))
    SIMMTM_DECODER_HIDDEN_DIM = int(os.environ.get('MEDAL_SIMMTM_DECODER_HIDDEN_DIM', 64))
    
    # SupCon parameters (Legacy - 保留兼容性)
    SUPCON_TEMPERATURE = 0.1
    SUPCON_LAMBDA = 1.0  # Weight for SupCon loss
    
    # ==================== Instance Contrastive Learning (New) ====================
    # 是否启用实例对比学习（替代原有的SupCon）
    USE_INSTANCE_CONTRASTIVE = True

    CONTRASTIVE_METHOD = os.getenv("MEDAL_CONTRASTIVE_METHOD", "infonce").lower()
    NNCLR_QUEUE_SIZE = 4096
    NNCLR_MIN_SIMILARITY = 0.0
    NNCLR_WARMUP_EPOCHS = 0
    
    # InfoNCE参数
    # 修复说明：调整温度和权重，优化对比学习收敛性
    # 温度从 0.3 增加到 0.5：让损失更平滑，更容易收敛到较低值
    # 权重保持 0.3：平衡 SimMTM 和 InfoNCE 的贡献
    INFONCE_TEMPERATURE = 0.5  # 温度系数τ（从0.3增到0.5，让损失更平滑易收敛）
    INFONCE_LAMBDA = 0.5  # InfoNCE损失权重（保持0.3，让SimMTM主导）
    
    # 流量增强参数
    # 修复说明：降低增强强度，避免破坏关键特征
    # 原因：过强的增强(80%裁剪)可能破坏恶意样本特征，导致Recall下降12.1%
    TRAFFIC_AUG_CROP_PROB = 0.5  # 时序裁剪概率（从0.8降到0.5）
    TRAFFIC_AUG_JITTER_PROB = 0.4  # 时序抖动概率（从0.6降到0.4）
    TRAFFIC_AUG_MASK_PROB = 0.3  # 通道掩码概率（从0.5降到0.3）
    TRAFFIC_AUG_CROP_MIN_RATIO = 0.5  # 最小裁剪比例
    TRAFFIC_AUG_CROP_MAX_RATIO = 0.9  # 最大裁剪比例
    TRAFFIC_AUG_JITTER_STD = 0.1  # 抖动标准差
    TRAFFIC_AUG_MASK_RATIO = 0.15  # 掩码比例
    
    # Burst 抖动增强（针对隧道流量的鲁棒性）
    # 模拟真实网络环境中的拥塞，导致突发大小和边界发生微小变化
    # 防止模型死记硬背具体的 Burst 数值
    TRAFFIC_AUG_BURST_JITTER_PROB = 0.5  # Burst 抖动概率
    TRAFFIC_AUG_BURST_JITTER_STD = 0.05  # Burst 抖动标准差（±5%）
    
    # ==================== Label Correction (Stage 2) ====================
    # CL (Confident Learning)
    CL_K_FOLD = 5
    
    # MADE (Density Estimation)
    MADE_HIDDEN_DIMS = [64, 128, 64]  # 降低到32维系统对应的隐藏层维度
    MADE_DENSITY_THRESHOLD_PERCENTILE = 70  # Top 70% = Dense
    
    # KNN (Semantic Voting)
    KNN_NEIGHBORS = 20
    KNN_METRIC = "euclidean"  # 距离度量方式: 'euclidean', 'manhattan', 'cosine', etc.
    
    # ==================== Data Augmentation (Stage 2) ====================
    # TabDDPM parameters
    STAGE2_USE_TABDDPM = True
    _env_stage2_tabddpm = os.environ.get('MEDAL_STAGE2_USE_TABDDPM', '').strip().lower()
    if _env_stage2_tabddpm in ('0', 'false', 'no', 'n', 'off'):
        STAGE2_USE_TABDDPM = False
    elif _env_stage2_tabddpm in ('1', 'true', 'yes', 'y', 'on'):
        STAGE2_USE_TABDDPM = True

    STAGE2_TABDDPM_SPACE = 'feature'

    STAGE2_FEATURE_AUG_MULTIPLIER = 5
    _env_stage2_feat_mult = os.environ.get('MEDAL_STAGE2_FEATURE_AUG_MULTIPLIER', '').strip()
    if _env_stage2_feat_mult:
        try:
            STAGE2_FEATURE_AUG_MULTIPLIER = int(float(_env_stage2_feat_mult))
        except ValueError:
            pass

    STAGE2_FEATURE_TIER1_MIN_WEIGHT = 0.9
    _env_stage2_feat_t1 = os.environ.get('MEDAL_STAGE2_FEATURE_TIER1_MIN_WEIGHT', '').strip()
    if _env_stage2_feat_t1:
        try:
            STAGE2_FEATURE_TIER1_MIN_WEIGHT = float(_env_stage2_feat_t1)
        except ValueError:
            pass
    STAGE2_FEATURE_TIER1_MULTIPLIER = 10
    _env_stage2_feat_t1m = os.environ.get('MEDAL_STAGE2_FEATURE_TIER1_MULTIPLIER', '').strip()
    if _env_stage2_feat_t1m:
        try:
            STAGE2_FEATURE_TIER1_MULTIPLIER = int(float(_env_stage2_feat_t1m))
        except ValueError:
            pass

    STAGE2_FEATURE_TIER2_MIN_WEIGHT = 0.7
    _env_stage2_feat_t2 = os.environ.get('MEDAL_STAGE2_FEATURE_TIER2_MIN_WEIGHT', '').strip()
    if _env_stage2_feat_t2:
        try:
            STAGE2_FEATURE_TIER2_MIN_WEIGHT = float(_env_stage2_feat_t2)
        except ValueError:
            pass
    STAGE2_FEATURE_TIER2_MULTIPLIER = 5
    _env_stage2_feat_t2m = os.environ.get('MEDAL_STAGE2_FEATURE_TIER2_MULTIPLIER', '').strip()
    if _env_stage2_feat_t2m:
        try:
            STAGE2_FEATURE_TIER2_MULTIPLIER = int(float(_env_stage2_feat_t2m))
        except ValueError:
            pass

    STAGE2_FEATURE_LOWCONF_MULTIPLIER = 0
    _env_stage2_feat_lowm = os.environ.get('MEDAL_STAGE2_FEATURE_LOWCONF_MULTIPLIER', '').strip()
    if _env_stage2_feat_lowm:
        try:
            STAGE2_FEATURE_LOWCONF_MULTIPLIER = int(float(_env_stage2_feat_lowm))
        except ValueError:
            pass

    DDPM_EPOCHS = 300
    DDPM_EARLY_STOPPING = True
    DDPM_ES_WARMUP_EPOCHS = 20
    DDPM_ES_PATIENCE = 30
    DDPM_ES_MIN_DELTA = 0.001
    DDPM_TIMESTEPS = 1000
    DDPM_HIDDEN_DIMS = [64, 128, 64]  # 降低到32维系统对应的隐藏层维度
    DDPM_SAMPLING_STEPS = 50  # DDIM sampling
    
    # Differential Guidance (Classifier-Free Guidance)
    # 引导强度设置原则：
    # - 良性流量：特征简单，无需强化 → w=1.0
    # - 恶意流量：特征多样，适度强化 → w=1.2
    # 过高的引导强度（如1.5+）会牺牲多样性和真实性
    GUIDANCE_BENIGN = 1.0        # 良性流量：无引导（保持自然分布）
    GUIDANCE_MALICIOUS = 1.2     # 恶意流量：适度强化（平衡质量与多样性）
    
    # Augmentation ratio
    # 生成倍数设置：
    # - MIN/MAX = 1: 每个样本生成1倍合成数据（总数据量翻倍）
    # - MIN/MAX = 2: 每个样本生成2倍合成数据（总数据量3倍）
    # 双通道生成：正负样本分开训练和生成，避免交叉污染
    AUGMENTATION_RATIO_MIN = 1
    AUGMENTATION_RATIO_MAX = 1
    
    # 增强策略模式
    # 'fixed': 固定倍数增强（所有样本统一使用 AUGMENTATION_RATIO_MIN）
    # 'weighted': 权重自适应增强（高权重样本多生成，低权重样本少生成或不生成）
    AUGMENTATION_MODE = 'fixed'  # 使用固定倍数模式
    
    # Structure-aware generation
    MASK_PROBABILITY = 0.5
    MASK_LAMBDA = 0.1
    
    # Feature indices for conditioning
    COND_FEATURE_INDICES = [2, 5]
    DEP_FEATURE_INDICES = [0, 1, 3, 4]

    ENABLE_COVARIANCE_MATCHING = False
    ENABLE_DISCRETE_QUANTIZATION = False
    DISCRETE_QUANTIZE_INDICES = []
    DISCRETE_QUANTIZE_MAX_VALUES = 4096
    AUGMENT_USE_WEIGHTED_SAMPLING = True
    
    # ==================== Classification (Stage 3) ====================
    # Dual-Stream MLP
    CLASSIFIER_HIDDEN_DIM = 32  # 降低到32维
    CLASSIFIER_OUTPUT_DIM = 2  # Binary classification
    
    # ==================== 温室训练+战场校准策略 ====================
    # Stage 3训练策略：强制1:1平衡采样（温室训练）
    USE_BALANCED_SAMPLING = True  # 启用WeightedRandomSampler强制1:1平衡
    
    # Validation set splitting for threshold optimization
    # Mixed validation set: 20% original + 10% synthetic
    # Prioritize real data to ensure threshold optimization reliability
    VALIDATION_SIZE_ORIGINAL = 0.2      # 20% of original data for validation
    VALIDATION_SIZE_SYNTHETIC = 0.1     # 10% of synthetic data for validation
    
    # Fine-tuning parameters
    # 增加训练轮数，配合早停机制
    FINETUNE_EPOCHS = 500  # 训练500轮（配合早停）
    FINETUNE_BATCH_SIZE = 128
    FINETUNE_LR = 1e-4
    FINETUNE_MIN_LR = 1e-6  # Minimum learning rate for cosine annealing
    TEST_BATCH_SIZE = 256
    
    # Stage 3: Early Stopping (智能训练终止)
    # 针对无验证集场景优化的早停策略
    FINETUNE_EARLY_STOPPING = True  # 启用早停机制
    FINETUNE_ES_WARMUP_EPOCHS = 100  # 前100轮不触发早停（让模型充分学习）
    FINETUNE_ES_PATIENCE = 50       # 50轮不改善则停止（更宽容）
    FINETUNE_ES_MIN_DELTA = 0.005   # F1改善阈值：0.5%（更敏感，适应波动）
    FINETUNE_ES_METRIC = 'f1_optimal'  # 监控指标：最优F1分数（自动阈值）
    FINETUNE_ES_ALLOW_TRAIN_METRIC = True  # 允许使用训练集指标（无验证集时）

    # Stage 3: Validation split for selecting best checkpoint (by val F1 pos=1 with auto-threshold)
    FINETUNE_VAL_SPLIT = 0.0  # 去掉验证集

    # Stage 3: Optional fixed validation set size per class (scheme A)
    # If >0, overrides FINETUNE_VAL_SPLIT and will reserve exactly N samples per class for validation.
    FINETUNE_VAL_PER_CLASS = 0
    _env_val_per_class = os.environ.get('MEDAL_FINETUNE_VAL_PER_CLASS', '').strip()
    if _env_val_per_class:
        try:
            FINETUNE_VAL_PER_CLASS = int(float(_env_val_per_class))
        except ValueError:
            pass

    MALICIOUS_THRESHOLD = 0.5

    # Stage 3: Online representation augmentation (input-level, physics-safe)
    STAGE3_ONLINE_AUGMENTATION = True
    _env_stage3_aug = os.environ.get('MEDAL_STAGE3_ONLINE_AUGMENTATION', '').strip().lower()
    if _env_stage3_aug in ('1', 'true', 'yes', 'y', 'on'):
        STAGE3_ONLINE_AUGMENTATION = True
    elif _env_stage3_aug in ('0', 'false', 'no', 'n', 'off'):
        STAGE3_ONLINE_AUGMENTATION = False

    STAGE3_MIXED_STREAM = False
    _env_stage3_mixed = os.environ.get('MEDAL_STAGE3_MIXED_STREAM', '').strip().lower()
    if _env_stage3_mixed in ('1', 'true', 'yes', 'y', 'on'):
        STAGE3_MIXED_STREAM = True
    elif _env_stage3_mixed in ('0', 'false', 'no', 'n', 'off'):
        STAGE3_MIXED_STREAM = False

    STAGE3_MIXED_REAL_BATCH_SIZE = 32
    _env_mixed_real_bs = os.environ.get('MEDAL_STAGE3_MIXED_REAL_BATCH_SIZE', '').strip()
    if _env_mixed_real_bs:
        try:
            STAGE3_MIXED_REAL_BATCH_SIZE = int(float(_env_mixed_real_bs))
        except ValueError:
            pass

    STAGE3_MIXED_SYN_BATCH_SIZE = 256
    _env_mixed_syn_bs = os.environ.get('MEDAL_STAGE3_MIXED_SYN_BATCH_SIZE', '').strip()
    if _env_mixed_syn_bs:
        try:
            STAGE3_MIXED_SYN_BATCH_SIZE = int(float(_env_mixed_syn_bs))
        except ValueError:
            pass

    STAGE3_MIXED_REAL_LOSS_SCALE = 1.0
    _env_mixed_real_scale = os.environ.get('MEDAL_STAGE3_MIXED_REAL_LOSS_SCALE', '').strip()
    if _env_mixed_real_scale:
        try:
            STAGE3_MIXED_REAL_LOSS_SCALE = float(_env_mixed_real_scale)
        except ValueError:
            pass
    
    # Stage 3: ST-Mixup (Spatio-Temporal Mixup) 增强
    # 渐进式类内混合，增强决策边界鲁棒性
    STAGE3_USE_ST_MIXUP = False  # 关闭ST-Mixup
    _env_st_mixup = os.environ.get('MEDAL_STAGE3_USE_ST_MIXUP', '').strip().lower()
    if _env_st_mixup in ('1', 'true', 'yes', 'y', 'on'):
        STAGE3_USE_ST_MIXUP = True
    elif _env_st_mixup in ('0', 'false', 'no', 'n', 'off'):
        STAGE3_USE_ST_MIXUP = False
    
    STAGE3_ST_MIXUP_MODE = 'intra_class'  # 'intra_class' 或 'selective'
    STAGE3_ST_MIXUP_ALPHA = 0.2  # Beta分布参数（越小混合越极端）
    STAGE3_ST_MIXUP_WARMUP_EPOCHS = 100  # 前100轮不启用（让模型先学基本边界）
    STAGE3_ST_MIXUP_MAX_PROB = 0.3  # 最大混合概率（30%的样本会被混合）
    STAGE3_ST_MIXUP_TIME_SHIFT_RATIO = 0.15  # 时间偏移比例（序列长度的15%）
    STAGE3_ST_MIXUP_UNCERTAINTY_THRESHOLD = 0.3  # 困难样本阈值（仅selective模式）

    STAGE3_HARD_MINING = False
    _env_hm = os.environ.get('MEDAL_STAGE3_HARD_MINING', '').strip().lower()
    if _env_hm in ('1', 'true', 'yes', 'y', 'on'):
        STAGE3_HARD_MINING = True
    elif _env_hm in ('0', 'false', 'no', 'n', 'off'):
        STAGE3_HARD_MINING = False
    STAGE3_HARD_MINING_WARMUP_EPOCHS = 5
    STAGE3_HARD_MINING_FREQ_EPOCHS = 3
    STAGE3_HARD_MINING_TOPK_RATIO = 0.2
    STAGE3_HARD_MINING_MULTIPLIER = 3.0
    STAGE3_HARD_MINING_POS_PROB_MAX = 0.70
    STAGE3_HARD_MINING_NEG_PROB_MIN = 0.60

    # Stage 3: Optional backbone fine-tuning
    # Default keeps the original design (frozen backbone) for stability.
    _env_ft = os.environ.get('MEDAL_FINETUNE_BACKBONE', '').strip().lower()
    if _env_ft in ('1', 'true', 'yes', 'y', 'on'):
        FINETUNE_BACKBONE = True
    elif _env_ft in ('0', 'false', 'no', 'n', 'off'):
        FINETUNE_BACKBONE = False
    else:
        FINETUNE_BACKBONE = False
    # Scope options:
    # - 'projection': only train the bidirectional projection head
    # - 'all': train the whole backbone
    _env_scope = os.environ.get('MEDAL_FINETUNE_BACKBONE_SCOPE', '').strip().lower()
    if _env_scope in ('projection', 'all'):
        FINETUNE_BACKBONE_SCOPE = _env_scope
    else:
        FINETUNE_BACKBONE_SCOPE = 'projection'
    # Use a smaller LR for backbone to avoid catastrophic forgetting
    _env_lr = os.environ.get('MEDAL_FINETUNE_BACKBONE_LR', '').strip()
    if _env_lr:
        try:
            FINETUNE_BACKBONE_LR = float(_env_lr)
        except ValueError:
            pass
    else:
        FINETUNE_BACKBONE_LR = 1e-5

    LABEL_SMOOTHING = 0.1
    CONSISTENCY_TEMPERATURE = 2.0
    CONSISTENCY_WARMUP_EPOCHS = 5
    
    # Dynamic loss weights (Linear schedule)
    # 降低正交约束权重，避免过度分离
    SOFT_ORTH_WEIGHT_START = 0.5   # 从 1.0 降到 0.5
    SOFT_ORTH_WEIGHT_END = 0.01
    CONSISTENCY_WEIGHT_START = 0.0
    CONSISTENCY_WEIGHT_END = 0.5   # 从 1.0 降到 0.5，减少强制一致性
    
    # Co-teaching
    CO_TEACHING_SELECT_RATE = 0.7  # Select top 70% low-loss samples

    USE_CO_TEACHING = False
    CO_TEACHING_WARMUP_EPOCHS = 10
    CO_TEACHING_MIN_SAMPLE_WEIGHT = 0.5
    
    # Class weights for imbalanced classification
    # RAPIER-style: 1:1 balance, let data speak for itself
    # After TabDDPM augmentation, classes are balanced, so equal weights are appropriate
    CLASS_WEIGHT_BENIGN = 1.0      # Weight for benign class (neutral)
    CLASS_WEIGHT_MALICIOUS = 1.0   # Weight for malicious class (neutral)
    
    # Decision threshold for malicious class
    # 根据最新 F1(pos=1) 扫描，最佳阈值约为 0.71
    MALICIOUS_THRESHOLD = 0.5       # 默认推理阈值（可按需调整）
    
    # ==================== Enhanced Loss Functions for Better Class Separation ====================
    # Focal Loss: focuses on hard examples to improve separation
    USE_FOCAL_LOSS = True          # 启用 Focal Loss（用于监督项）
    # 修改：alpha=0.5 表示两类平等，避免对恶意类过度偏好
    # 原 alpha=0.25 导致恶意类权重是良性类的 3 倍，造成过度分离
    FOCAL_ALPHA = 0.5              # Alpha for malicious class (改为 0.5，两类平等)
    _env_focal_alpha = os.environ.get('MEDAL_FOCAL_ALPHA', '').strip()
    if _env_focal_alpha:
        try:
            FOCAL_ALPHA = float(_env_focal_alpha)
        except ValueError:
            pass
    FOCAL_GAMMA = 2.0              # Gamma parameter (focus on hard examples)
    
    # Soft F1 Loss: 直接优化 Binary F1-Score
    # ⚠️ 注意：在 1:1 平衡训练集上，Soft F1 Loss 可能导致过度分离
    # 降低权重以减少对恶意类的过度偏好
    USE_SOFT_F1_LOSS = False        # 启用 Soft F1 Loss
    SOFT_F1_WEIGHT = 0.1           # 从 0.5 降到 0.3，减少 F1 Loss 的影响

    USE_LOGIT_MARGIN = True
    _env_logit_margin = os.environ.get('MEDAL_USE_LOGIT_MARGIN', '').strip().lower()
    if _env_logit_margin in ('1', 'true', 'yes', 'y', 'on'):
        USE_LOGIT_MARGIN = True
    elif _env_logit_margin in ('0', 'false', 'no', 'n', 'off'):
        USE_LOGIT_MARGIN = False
    LOGIT_MARGIN_M = 0.25
    _env_logit_m = os.environ.get('MEDAL_LOGIT_MARGIN_M', '').strip()
    if _env_logit_m:
        try:
            LOGIT_MARGIN_M = float(_env_logit_m)
        except ValueError:
            pass
    LOGIT_MARGIN_WARMUP_EPOCHS = 30
    _env_logit_warm = os.environ.get('MEDAL_LOGIT_MARGIN_WARMUP_EPOCHS', '').strip()
    if _env_logit_warm:
        try:
            LOGIT_MARGIN_WARMUP_EPOCHS = int(float(_env_logit_warm))
        except ValueError:
            pass
    
    # Margin Loss (ArcFace-style): adds margin between classes
    # Disabled initially to avoid probability distortion (can re-enable after model stabilizes)
    USE_MARGIN_LOSS = False        # Disabled: avoid probability distortion during initial training
    MARGIN_M = 0.15                # Margin value (conservative)
    MARGIN_S = 1.0                 # Scale factor (set to 1.0 to avoid logit compression)
    MARGIN_LOSS_WEIGHT = 0.15      # Weight for margin loss (if re-enabled)
    MARGIN_LOSS_WEIGHT_START = 0.0 # Start at 0 (no margin loss in early training)
    MARGIN_LOSS_WEIGHT_END = 0.2   # End at 0.2 (if re-enabled)
    
    # Label Smoothing: reduces overconfidence
    LABEL_SMOOTHING = 0.05         # Label smoothing factor (reduced from 0.1)

    # ==================== Hybrid Court 标签矫正参数 ====================
    # ========== 三阶段标签矫正策略 v9 ==========
    # Phase 1: 核心严选 (Core Selection) - CL or KNN 筛选核心数据
    # Phase 2: 分级挽救 (Rescue & Tiering) - 翻转/保持/重加权/丢弃
    # Phase 3: 锚点拯救 (Anchor Rescue) - 以Core+Flip+Keep-High为基准进行新一轮数据拯救
    
    # ========== Phase 1: 核心严选阈值 (基于微幅保守策略 +0.02) ==========
    # CL置信度下限
    HC_PHASE1_CL_BENIGN = 0.54      # 正常样本CL置信度阈值
    HC_PHASE1_CL_MALICIOUS = 0.57   # 恶意样本CL置信度阈值
    # KNN一致性下限
    HC_PHASE1_KNN_BENIGN = 0.60     # 正常样本KNN一致性阈值
    HC_PHASE1_KNN_MALICIOUS = 0.70  # 恶意样本KNN一致性阈值
    
    # ========== Phase 2: 分级挽救阈值 ==========
    HC_PHASE2_REWEIGHT_BASE_CL = 0.35   # 重加权的基础准入线
    HC_PHASE2_MADE_ANOMALY = 60.0       # MADE密度异常分界线(用于Keep组)
    HC_PHASE2_SYS_CONF_SPLIT = 0.30     # 系统置信度分界线(用于Reweight组)
    
    # ========== Phase 3: 锚点拯救阈值 ==========
    HC_PHASE3_MIN_ANCHORS = 20          # 最少锚点样本数
    HC_PHASE3_KNN_K = 15                # 锚点KNN的K值
    HC_PHASE3_RESCUE_KEEP_CONS = 0.70   # 拯救为Keep的最低一致性
    HC_PHASE3_RESCUE_FLIP_CONS = 0.75   # 拯救为Flip的最低一致性
    HC_PHASE3_RESCUE_KEEP_WEIGHT = 0.85 # 拯救Keep的权重
    HC_PHASE3_RESCUE_FLIP_WEIGHT = 0.75 # 拯救Flip的权重
    
    # ========== 权重分配 (用户定制版) ==========
    HC_WEIGHT_TIER1_CORE = 1.0      # Tier 1: 核心样本
    HC_WEIGHT_TIER2_FLIP = 1.0      # Tier 2: 翻转样本
    HC_WEIGHT_TIER3A_KEEP_HI = 1.0  # Tier 3a: 优质保持
    HC_WEIGHT_TIER3B_KEEP_LO = 0.4  # Tier 3b: 存疑保持
    HC_WEIGHT_TIER4A_REW_HI = 0.6   # Tier 4a: 优质重加权
    HC_WEIGHT_TIER4B_REW_LO = 0.1   # Tier 4b: 噪声重加权
    
    # ========== 兼容旧版配置 (保留但不再使用) ==========
    # 以下参数保留以兼容旧代码，新代码使用上述三阶段配置
    HC_PREKEEP_BENIGN_MIN_KNN_CONSISTENCY = 0.60
    HC_PREKEEP_BENIGN_MIN_CL_PROB = 0.55
    HC_PREKEEP_MALICIOUS_MIN_KNN_CONSISTENCY = 0.50
    HC_PREKEEP_MALICIOUS_MIN_CL_PROB = 0.50
    HC_PREKEEP_REQUIRE_DENSE = False

    HC_CORELITE_WEIGHT = 0.8
    HC_CORELITE_BENIGN_MIN_KNN_CONSISTENCY = 0.55
    HC_CORELITE_MALICIOUS_MIN_KNN_CONSISTENCY = 0.45
    HC_CORELITE_MIN_CL_PROB = 0.50
    HC_CORELITE_REQUIRE_DENSE = False

    HC_FLIP_MIN_KNN_CONSISTENCY = 0.0
    HC_FLIP_MIN_CL_MAX_PROB = 0.0
    HC_FLIP_BENIGN_TO_MAL_MIN_KNN = 0.60
    HC_FLIP_BENIGN_TO_MAL_MIN_CL = 0.55
    HC_FLIP_MAL_TO_BENIGN_MIN_KNN = 0.55
    HC_FLIP_MAL_TO_BENIGN_MIN_CL = 0.50
    HC_FLIP_MAL_TO_BENIGN_MIN_DENSITY = 10.0

    HC_REWEIGHT_WEIGHT_SUSPECTED = 0.5
    HC_REWEIGHT_MIN_KNN_CONSISTENCY = 0.5
    HC_REWEIGHT_WEIGHT_LOWCONS = 0.3
    HC_REWEIGHT_WEIGHT_LONGTAIL = 0.7
    HC_MALICIOUS_PROTECT_WEIGHT = 0.4

    HC_MALICIOUS_LOW_DENSITY_DROP_MODE = "score"
    HC_MALICIOUS_LOW_DENSITY_DROP_PERCENTILE = 3.0
    HC_MALICIOUS_LOW_DENSITY_DROP_SCORE = -50.0
    HC_BENIGN_LOW_DENSITY_DROP_SCORE = -30.0

    HC_REVIEW_ENABLE = False
    HC_REVIEW_REVERT_MIN_KNN = 0.75
    HC_REVIEW_REVERT_MIN_CL = 0.75

    HC_ENABLE_SECOND_ROUND = True
    HC_SECOND_ROUND_MIN_ANCHORS = 20
    HC_SECOND_ROUND_ANCHOR_MIN_WEIGHT = 0.9
    HC_SECOND_ROUND_KEEP_MIN_CONS = 0.75
    HC_SECOND_ROUND_FLIP_MIN_CONS = 0.80

    # 数据增强模板样本质量门槛
    # 只有高质量样本才能作为生成模板，避免低质量样本污染合成数据
    AUGMENT_TEMPLATE_MIN_WEIGHT = 0.7   # 从 0.5 提高到 0.7，只用高质量样本
    AUGMENT_TEMPLATE_MIN_WEIGHT_HARD = 0.5  # 从 0.2 提高到 0.5，硬门槛也提高

    # ==================== Label Correction (HybridCourt) ====================
    HYBRIDCOURT_DYNAMIC_DENSITY_THRESHOLDS = False
    _env_hc_dyn = os.environ.get('MEDAL_HYBRIDCOURT_DYNAMIC_DENSITY_THRESHOLDS', '').strip().lower()
    if _env_hc_dyn in ('1', 'true', 'yes', 'y', 'on'):
        HYBRIDCOURT_DYNAMIC_DENSITY_THRESHOLDS = True
    elif _env_hc_dyn in ('0', 'false', 'no', 'n', 'off'):
        HYBRIDCOURT_DYNAMIC_DENSITY_THRESHOLDS = False

    HYBRIDCOURT_DENSITY_HIGH_PCT = float(os.environ.get('MEDAL_HYBRIDCOURT_DENSITY_HIGH_PCT', 90))
    HYBRIDCOURT_DENSITY_LOW_PCT = float(os.environ.get('MEDAL_HYBRIDCOURT_DENSITY_LOW_PCT', 50))

    # ==================== Classifier Input ====================
    CLASSIFIER_INPUT_IS_FEATURES = False
    _env_cls_feat = os.environ.get('MEDAL_CLASSIFIER_INPUT_IS_FEATURES', '').strip().lower()
    if _env_cls_feat in ('1', 'true', 'yes', 'y', 'on'):
        CLASSIFIER_INPUT_IS_FEATURES = True
    elif _env_cls_feat in ('0', 'false', 'no', 'n', 'off'):
        CLASSIFIER_INPUT_IS_FEATURES = False
    
    # ==================== Hardware & Training ====================
    GPU_ID = int(os.environ.get('MEDAL_GPU_ID', '7'))  # 默认使用GPU 7
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        # If CUDA_VISIBLE_DEVICES is set, device indices are remapped to 0..N-1.
        # Also guard against out-of-range GPU_ID to avoid "invalid device ordinal".
        _visible_count = torch.cuda.device_count()
        _safe_gpu_id = GPU_ID if 0 <= int(GPU_ID) < _visible_count else 0
        DEVICE = torch.device(f"cuda:{_safe_gpu_id}")
    else:
        DEVICE = torch.device("cpu")
    NUM_WORKERS = 4
    SEED = 42
    
    # Learning rate scheduler
    LR_SCHEDULER = "cosine"  # 'cosine' or 'step'
    
    # ==================== Visualization ====================
    VIS_FEATURE_DIM_REDUCTION = "tsne"  # 'tsne' or 'pca'
    VIS_PERPLEXITY = 30
    VIS_N_ITER = 1000
    
    # ==================== Output Directory Structure ====================
    # Module-specific output directories
    PREPROCESSING_DIR = os.path.join(OUTPUT_ROOT, "preprocessing")
    FEATURE_EXTRACTION_DIR = os.path.join(OUTPUT_ROOT, "feature_extraction")
    LABEL_CORRECTION_DIR = os.path.join(OUTPUT_ROOT, "label_correction")
    DATA_AUGMENTATION_DIR = os.path.join(OUTPUT_ROOT, "data_augmentation")
    CLASSIFICATION_DIR = os.path.join(OUTPUT_ROOT, "classification")
    RESULT_DIR = os.path.join(OUTPUT_ROOT, "result")
    
    # Legacy checkpoint directory (deprecated, use module-specific dirs)
    CHECKPOINT_DIR = CLASSIFICATION_DIR  # For backward compatibility
    
    # Save frequency
    SAVE_EVERY_N_EPOCHS = 10
    
    @staticmethod
    def create_dirs():
        """Create necessary directories"""
        os.makedirs(Config.OUTPUT_ROOT, exist_ok=True)
        
        # Create module-specific directories
        os.makedirs(Config.PREPROCESSING_DIR, exist_ok=True)
        os.makedirs(Config.FEATURE_EXTRACTION_DIR, exist_ok=True)
        os.makedirs(Config.LABEL_CORRECTION_DIR, exist_ok=True)
        os.makedirs(Config.DATA_AUGMENTATION_DIR, exist_ok=True)
        os.makedirs(Config.CLASSIFICATION_DIR, exist_ok=True)
        os.makedirs(Config.RESULT_DIR, exist_ok=True)
        
        # Create subdirectories for each module
        for module_dir in [Config.PREPROCESSING_DIR, Config.FEATURE_EXTRACTION_DIR,
                          Config.LABEL_CORRECTION_DIR, Config.DATA_AUGMENTATION_DIR,
                          Config.CLASSIFICATION_DIR, Config.RESULT_DIR]:
            os.makedirs(os.path.join(module_dir, "models"), exist_ok=True)
            os.makedirs(os.path.join(module_dir, "figures"), exist_ok=True)
            os.makedirs(os.path.join(module_dir, "logs"), exist_ok=True)
    
    @staticmethod
    def get_dynamic_weight(epoch, total_epochs, start_weight, end_weight):
        """Linear schedule for dynamic weights"""
        return start_weight + (end_weight - start_weight) * (epoch / total_epochs)
    
    def __repr__(self):
        """Print all configuration parameters"""
        config_str = "\n" + "="*50 + "\n"
        config_str += "MEDAL-Lite Configuration\n"
        config_str += "="*50 + "\n"
        
        for attr in dir(self):
            if not attr.startswith('_') and attr.isupper():
                config_str += f"{attr}: {getattr(self, attr)}\n"
        
        config_str += "="*50 + "\n"
        return config_str
    
    def print_config(self):
        """打印配置信息"""
        print(self)
    
    def get_all_configs_dict(self):
        """获取所有配置的字典表示"""
        config_dict = {}
        for attr in dir(self):
            if not attr.startswith('_') and attr.isupper():
                config_dict[attr] = getattr(self, attr)
        return config_dict


# Global config instance
config = Config()


# 便捷函数
def print_config():
    """打印配置信息 (便捷函数)"""
    config.print_config()


if __name__ == "__main__":
    # 测试配置加载
    print_config()
    print(f"\n✓ 项目根目录: {config.PROJECT_ROOT}")
    print(f"✓ 数据根目录: {config.DATA_ROOT}")
    print(f"✓ 输出根目录: {config.OUTPUT_ROOT}")
