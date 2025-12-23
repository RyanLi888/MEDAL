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
    DATA_ROOT = os.path.join(PROJECT_ROOT, "Datasets")
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
    
    # 5维特征
    FEATURE_NAMES = ['Length', 'Log-IAT', 'Direction', 'Flags', 'Window']
    
    # 特征归一化参数
    MTU = 1500.0  # 最大传输单元
    TCP_WINDOW_MAX = 65535.0  # TCP窗口最大值
    IAT_EPSILON = 1e-7  # Log-IAT 防止log(0)
    
    # ==================== Input & Embedding Parameters ====================
    SEQUENCE_LENGTH = 1024  # L: Maximum number of packets per flow
    INPUT_FEATURE_DIM = 5   # [Length, Log-IAT, Direction, Flags, Window]
    MODEL_DIM = 64          # d_model: Embedding dimension
    EMBEDDING_DROPOUT = 0.1
    POSITIONAL_ENCODING = "sinusoidal"  # 正弦位置编码
    
    # ==================== Micro-Bi-Mamba Backbone ====================
    MAMBA_LAYERS = 2
    MAMBA_STATE_DIM = 16        # d_state: SSM internal memory capacity
    MAMBA_EXPANSION_FACTOR = 2   # E: Expansion factor
    MAMBA_CONV_KERNEL = 4        # Local conv kernel size
    MAMBA_DROPOUT = 0.1
    MAMBA_FUSION_TYPE = "concat_project"  # "average" 或 "concat_project"
    MAMBA_PROJECTION_DIM = 128   # Concat后的投影维度(64*2)
    
    # ==================== Pre-training (Stage 1) ====================
    PRETRAIN_EPOCHS = 200
    PRETRAIN_BATCH_SIZE = 64
    PRETRAIN_LR = 1e-3
    PRETRAIN_WEIGHT_DECAY = 1e-4
    
    # SimMTM parameters
    SIMMTM_MASK_RATE = 0.5  # 50% masking
    
    # SupCon parameters
    SUPCON_TEMPERATURE = 0.1
    SUPCON_LAMBDA = 1.0  # Weight for SupCon loss
    
    # ==================== Label Correction (Stage 2) ====================
    # CL (Confident Learning)
    CL_K_FOLD = 5
    
    # MADE (Density Estimation)
    MADE_HIDDEN_DIMS = [128, 256, 128]
    MADE_DENSITY_THRESHOLD_PERCENTILE = 70  # Top 70% = Dense
    
    # KNN (Semantic Voting)
    KNN_NEIGHBORS = 20
    KNN_METRIC = "euclidean"  # 距离度量方式: 'euclidean', 'manhattan', 'cosine', etc.
    
    # ==================== Data Augmentation (Stage 2) ====================
    # TabDDPM parameters
    DDPM_TIMESTEPS = 1000
    DDPM_HIDDEN_DIMS = [128, 256, 128]
    DDPM_SAMPLING_STEPS = 50  # DDIM sampling
    
    # Differential Guidance
    # Balanced guidance to generate realistic samples with good diversity
    # High guidance (2.5) generates overly "typical" samples that are far from decision boundary
    # Low guidance (<1.0) may generate samples too close to noise
    GUIDANCE_MALICIOUS = 1.5  # Moderate guidance (balanced: good diversity + quality)
    GUIDANCE_BENIGN = 1.2     # Slightly higher guidance to ensure quality
    
    # Augmentation ratio
    # ⚠️ 问题：过多的合成数据导致分布偏移，Benign 被推向 Malicious 区域
    # 从 8x 降到 2x，减少合成数据的影响
    AUGMENTATION_RATIO_MIN = 2
    AUGMENTATION_RATIO_MAX = 2
    
    # Structure-aware generation
    MASK_PROBABILITY = 0.5
    MASK_LAMBDA = 0.1
    
    # Feature indices for conditioning
    COND_FEATURE_INDICES = [2, 3]  # Direction, Flags
    DEP_FEATURE_INDICES = [0, 1, 4]  # Length, Log-IAT, Window
    
    # ==================== Classification (Stage 3) ====================
    # Dual-Stream MLP
    CLASSIFIER_HIDDEN_DIM = 64
    CLASSIFIER_OUTPUT_DIM = 2  # Binary classification
    
    # Validation set splitting for threshold optimization
    # Mixed validation set: 20% original + 10% synthetic
    # Prioritize real data to ensure threshold optimization reliability
    VALIDATION_SIZE_ORIGINAL = 0.2      # 20% of original data for validation
    VALIDATION_SIZE_SYNTHETIC = 0.1     # 10% of synthetic data for validation
    
    # Fine-tuning parameters
    # Increased from 50 to 80 to allow Focal Loss and Margin Loss to fully converge
    # Margin Loss weight increases from 0.3 to 0.7, needs more epochs to reach full effect
    FINETUNE_EPOCHS = 80
    FINETUNE_BATCH_SIZE = 64
    FINETUNE_LR = 1e-4

    # Stage 3: Optional backbone fine-tuning
    # Default keeps the original design (frozen backbone) for stability.
    FINETUNE_BACKBONE = False
    # Scope options:
    # - 'projection': only train the bidirectional projection head
    # - 'all': train the whole backbone
    FINETUNE_BACKBONE_SCOPE = 'projection'
    # Use a smaller LR for backbone to avoid catastrophic forgetting
    FINETUNE_BACKBONE_LR = 1e-5
    
    # Dynamic loss weights (Linear schedule)
    # 降低正交约束权重，避免过度分离
    SOFT_ORTH_WEIGHT_START = 0.5   # 从 1.0 降到 0.5
    SOFT_ORTH_WEIGHT_END = 0.01
    CONSISTENCY_WEIGHT_START = 0.0
    CONSISTENCY_WEIGHT_END = 0.5   # 从 1.0 降到 0.5，减少强制一致性
    
    # Co-teaching
    CO_TEACHING_SELECT_RATE = 0.7  # Select top 70% low-loss samples

    USE_CO_TEACHING = True
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
    FOCAL_GAMMA = 2.0              # Gamma parameter (focus on hard examples)
    
    # Soft F1 Loss: 直接优化 Binary F1-Score
    # ⚠️ 注意：在 1:1 平衡训练集上，Soft F1 Loss 可能导致过度分离
    # 降低权重以减少对恶意类的过度偏好
    USE_SOFT_F1_LOSS = True        # 启用 Soft F1 Loss
    SOFT_F1_WEIGHT = 0.3           # 从 0.5 降到 0.3，减少 F1 Loss 的影响
    
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
    
    # ==================== Hardware & Training ====================
    GPU_ID = 7  # 默认使用GPU 7
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
