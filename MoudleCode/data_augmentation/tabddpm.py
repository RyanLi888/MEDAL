"""
TabDDPM: Tabular Denoising Diffusion Probabilistic Model
For structure-aware data augmentation

核心创新：
1. 协议感知去噪 (Protocol-Aware Denoising)
   - 标准DDPM: 仅学习去噪 ε_pred ≈ ε_true
   - 本实现: 去噪 + 结构重建 (Structure-Aware Masking)
   - 损失函数: L_total = L_diffusion + λ * L_reconstruction
   
2. 分类器自由引导 (Classifier-Free Guidance, CFG)
   - 训练时: 20% 概率丢弃标签 (y=-1)，学习无条件生成
   - 生成时: noise_pred = noise_uncond + w * (noise_cond - noise_uncond)
   - 引导强度: w=1.5 (恶意), w=1.2 (良性) - 控制类别特征强度
   
3. 条件-依赖解耦 (Condition-Dependence Decoupling)
   - 条件特征 (固定): Direction, ValidMask - 协议约束
   - 依赖特征 (生成): Length, BurstSize - 学习分布
   - 训练时随机mask依赖特征，强制模型学习特征间依赖关系

训练目标：
- 输入: x_t (噪声数据) + t (时间步) + y (标签) + x_cond (条件特征)
- 输出: ε_pred (预测噪声)
- 优化: ||ε_pred - ε_true||² + λ * ||ε_pred[masked] - ε_true[masked]||²

生成质量评估：
1. Fidelity (真实性): 特征分布统计距离 (均值/方差对比)
2. Diversity (多样性): 最近邻距离 (防止记忆训练集)
3. Utility (实用性): TSTR (Train on Synthetic, Test on Real) F1-Score
4. Protocol Validity (协议有效性): 物理约束检查 (无负数、范围合法)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ResidualMLP(nn.Module):
    """Residual MLP for denoising network"""
    
    def __init__(self, input_dim=5, hidden_dims=[128, 256, 128], output_dim=5):
        super().__init__()
        
        # Log initialization for debugging
        logger.info(f"ResidualMLP initialization:")
        logger.info(f"  input_dim: {input_dim}")
        logger.info(f"  hidden_dims: {hidden_dims}")
        logger.info(f"  output_dim: {output_dim}")
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            logger.info(f"  Layer {i}: Linear({prev_dim}, {hidden_dim})")
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        logger.info(f"  Final layer: Linear({prev_dim}, {output_dim})")
        
        self.network = nn.Sequential(*layers)
        
        # Skip connection if dimensions match
        self.use_skip = (input_dim == output_dim)
    
    def forward(self, x):
        # Debug: check input dimensions
        expected_dim = self.network[0].in_features
        actual_dim = x.shape[-1]
        
        if actual_dim != expected_dim:
            logger.error(f"❌ ResidualMLP input dimension mismatch:")
            logger.error(f"  网络期望输入维度: {expected_dim}")
            logger.error(f"  实际输入维度: {actual_dim}")
            logger.error(f"  输入形状: {x.shape}")
            logger.error(f"  第一层权重形状: {self.network[0].weight.shape}")
            logger.error(f"  第一层权重: (out_features={self.network[0].out_features}, in_features={self.network[0].in_features})")
            raise ValueError(f"Input dimension mismatch: expected {expected_dim}, got {actual_dim}")
        
        out = self.network(x)
        if self.use_skip:
            out = out + x
        return out


class TabDDPM(nn.Module):
    """
    Tabular Denoising Diffusion Probabilistic Model
    
    Operates on configurable-dimensional raw features with protocol-aware generation
    """
    
    def __init__(self, config, input_dim=None, cond_indices=None, dep_indices=None, enable_protocol_constraints=True):
        super().__init__()
        
        self.config = config
        self.input_dim = int(input_dim if input_dim is not None else config.INPUT_FEATURE_DIM)
        self.timesteps = config.DDPM_TIMESTEPS
        self.enable_protocol_constraints = bool(enable_protocol_constraints)
        
        # Condition and dependence feature indices (must be defined before denoising_net)
        self.cond_indices = list(cond_indices) if cond_indices is not None else list(config.COND_FEATURE_INDICES)
        self.dep_indices = list(dep_indices) if dep_indices is not None else list(config.DEP_FEATURE_INDICES)
        
        # Denoising network
        # Input: x_t (D) + t (1) + y (1) + x_cond (len(cond_indices))
        cond_dim = len(self.cond_indices)  # Number of conditional features
        denoising_input_dim = self.input_dim + 1 + 1 + cond_dim  # x + t + y + cond_features
        
        # Log dimension for debugging
        logger.info(f"TabDDPM initialization:")
        logger.info(f"  input_dim: {self.input_dim}")
        logger.info(f"  cond_indices: {self.cond_indices} (dim={cond_dim})")
        logger.info(f"  denoising_net input_dim: {denoising_input_dim} ({self.input_dim} + 1 + 1 + {cond_dim})")
        
        self.denoising_net = ResidualMLP(
            input_dim=denoising_input_dim,
            hidden_dims=config.DDPM_HIDDEN_DIMS,
            output_dim=self.input_dim
        )
        
        # Noise schedule (beta)
        self.register_buffer('betas', self._cosine_beta_schedule())
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                             torch.sqrt(1.0 - self.alphas_cumprod))

        self.register_buffer('feature_mean', torch.zeros(self.input_dim))
        self.register_buffer('feature_std', torch.ones(self.input_dim))
        self.register_buffer('raw_min', torch.full((self.input_dim,), float('-inf')))
        self.register_buffer('raw_max', torch.full((self.input_dim,), float('inf')))
        self.register_buffer('scaler_fitted', torch.tensor(0, dtype=torch.uint8))
    
    def _cosine_beta_schedule(self, s=0.008):
        """Cosine noise schedule"""
        steps = self.timesteps + 1
        t = torch.linspace(0, self.timesteps, steps)
        alphas_cumprod = torch.cos(((t / self.timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def fit_scaler(self, X_raw, clip_percentiles=(0.5, 99.5)):
        X_raw = np.asarray(X_raw, dtype=np.float32)
        if X_raw.ndim != 2 or X_raw.shape[1] != self.input_dim:
            raise ValueError(f"X_raw must have shape (N, {self.input_dim}), got {X_raw.shape}")

        mean = X_raw.mean(axis=0)
        std = X_raw.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)

        p_low, p_high = clip_percentiles
        raw_min = np.percentile(X_raw, p_low, axis=0)
        raw_max = np.percentile(X_raw, p_high, axis=0)

        # Protocol-aligned hard ranges (keep others by percentile)
        if self.enable_protocol_constraints:
            length_idx = getattr(self.config, 'LENGTH_INDEX', None)
            if length_idx is not None:
                length_idx = int(length_idx)
                if 0 <= length_idx < self.input_dim:
                    raw_min[length_idx] = 0.0
                    raw_max[length_idx] = 1.0

            direction_idx = getattr(self.config, 'DIRECTION_INDEX', None)
            if direction_idx is not None:
                direction_idx = int(direction_idx)
                if 0 <= direction_idx < self.input_dim:
                    raw_min[direction_idx] = -1.0
                    raw_max[direction_idx] = 1.0

            burst_idx = getattr(self.config, 'BURST_SIZE_INDEX', None)
            if burst_idx is not None:
                burst_idx = int(burst_idx)
                if 0 <= burst_idx < self.input_dim:
                    raw_min[burst_idx] = max(float(raw_min[burst_idx]), 0.0)

            valid_mask_idx = getattr(self.config, 'VALID_MASK_INDEX', None)
            if valid_mask_idx is not None:
                valid_mask_idx = int(valid_mask_idx)
                if 0 <= valid_mask_idx < self.input_dim:
                    raw_min[valid_mask_idx] = 0.0
                    raw_max[valid_mask_idx] = 1.0

        device = self.feature_mean.device
        self.feature_mean.data.copy_(torch.from_numpy(mean).to(device))
        self.feature_std.data.copy_(torch.from_numpy(std).to(device))
        self.raw_min.data.copy_(torch.from_numpy(raw_min).to(device))
        self.raw_max.data.copy_(torch.from_numpy(raw_max).to(device))
        self.scaler_fitted.data.fill_(1)

    def transform(self, x_raw):
        if int(self.scaler_fitted.item()) != 1:
            raise RuntimeError("TabDDPM scaler not fitted. Call fit_scaler(...) before training/sampling.")
        return (x_raw - self.feature_mean) / self.feature_std

    def inverse_transform(self, x_scaled):
        if int(self.scaler_fitted.item()) != 1:
            raise RuntimeError("TabDDPM scaler not fitted. Call fit_scaler(...) before training/sampling.")
        return x_scaled * self.feature_std + self.feature_mean

    def project_protocol_constraints(self, x_raw):
        x = x_raw
        if self.enable_protocol_constraints:
            length_idx = getattr(self.config, 'LENGTH_INDEX', None)
            if length_idx is not None:
                length_idx = int(length_idx)
                if 0 <= length_idx < self.input_dim:
                    x[..., length_idx] = torch.clamp(x[..., length_idx], 0.0, 1.0)

            direction_idx = getattr(self.config, 'DIRECTION_INDEX', None)
            if direction_idx is not None:
                direction_idx = int(direction_idx)
                if 0 <= direction_idx < self.input_dim:
                    x[..., direction_idx] = torch.clamp(x[..., direction_idx], -1.0, 1.0)

            burst_idx = getattr(self.config, 'BURST_SIZE_INDEX', None)
            if burst_idx is not None:
                burst_idx = int(burst_idx)
                if 0 <= burst_idx < self.input_dim:
                    x[..., burst_idx] = torch.clamp(x[..., burst_idx], 0.0, float('inf'))

            valid_mask_idx = getattr(self.config, 'VALID_MASK_INDEX', None)
            if valid_mask_idx is not None:
                valid_mask_idx = int(valid_mask_idx)
                if 0 <= valid_mask_idx < self.input_dim:
                    x[..., valid_mask_idx] = torch.clamp(x[..., valid_mask_idx], 0.0, 1.0)

        x = torch.max(torch.min(x, self.raw_max), self.raw_min)
        return x

    @staticmethod
    def _topk_discrete_values(values, max_values):
        values = np.asarray(values)
        if values.size == 0:
            return np.asarray([], dtype=np.float32)
        uniq, cnt = np.unique(values.astype(np.float32), return_counts=True)
        if uniq.size > max_values:
            idx = np.argpartition(cnt, -max_values)[-max_values:]
            uniq = uniq[idx]
        uniq = np.sort(uniq)
        return uniq

    @staticmethod
    def _quantize_to_vocab(x, vocab):
        x = np.asarray(x, dtype=np.float32)
        vocab = np.asarray(vocab, dtype=np.float32)
        if vocab.size == 0:
            return x
        flat = x.reshape(-1)
        pos = np.searchsorted(vocab, flat, side='left')
        pos0 = np.clip(pos - 1, 0, vocab.size - 1)
        pos1 = np.clip(pos, 0, vocab.size - 1)
        v0 = vocab[pos0]
        v1 = vocab[pos1]
        choose1 = np.abs(flat - v1) < np.abs(flat - v0)
        out = np.where(choose1, v1, v0)
        return out.reshape(x.shape)
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: add noise to x_0
        
        Args:
            x_0: (B, D) - clean data
            t: (B,) - timesteps
            noise: (B, D) - noise (optional)
            
        Returns:
            x_t: (B, D) - noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t
    
    def predict_noise(self, x_t, t, y, x_cond=None):
        """
        Predict noise using denoising network
        
        Args:
            x_t: (B, D) - noisy data
            t: (B,) - timesteps
            y: (B,) - labels
            x_cond: (B, len(cond_indices)) - conditional features
            
        Returns:
            predicted_noise: (B, D)
        """
        # Normalize timestep
        t_norm = t.float() / self.timesteps
        
        if x_cond is None:
            x_cond = torch.empty((x_t.shape[0], 0), device=x_t.device, dtype=x_t.dtype)

        # Concatenate inputs
        net_input = torch.cat([
            x_t,  # (B, D)
            t_norm.unsqueeze(-1),  # (B, 1)
            y.unsqueeze(-1).float(),  # (B, 1)
            x_cond  # (B, cond_dim)
        ], dim=-1)
        
        # Debug: check input dimensions
        expected_dim = self.input_dim + 1 + 1 + len(self.cond_indices)
        if net_input.shape[-1] != expected_dim:
            logger.error(f"Dimension mismatch in predict_noise:")
            logger.error(f"  Expected input dim: {expected_dim}")
            logger.error(f"  Actual input dim: {net_input.shape[-1]}")
            logger.error(f"  x_t shape: {x_t.shape}")
            logger.error(f"  x_cond shape: {x_cond.shape}")
            logger.error(f"  cond_indices: {self.cond_indices}")
            raise ValueError(f"Input dimension mismatch: expected {expected_dim}, got {net_input.shape[-1]}")
        
        predicted_noise = self.denoising_net(net_input)
        
        return predicted_noise
    
    def compute_loss(self, x_0, y, mask_prob=0.5, mask_lambda=0.1, p_uncond=0.2):
        """
        Compute training loss with structure-aware masking and classifier-free guidance training
        
        核心机制详解：
        
        1. 标准DDPM损失 (Diffusion Loss):
           L_diff = ||ε_pred - ε_true||²
           - 训练去噪网络预测噪声
           - 每个样本每次训练都加不同的随机噪声 → 天然抗过拟合
        
        2. 结构感知重建损失 (Structure-Aware Reconstruction Loss):
           L_recon = ||ε_pred[masked] - ε_true[masked]||²
           - 随机mask依赖特征 (Length, BurstSize)
           - 强制模型学习特征间的依赖关系
           - 例如: 短握手包通常对应较小的同向BurstSize
        
        3. 分类器自由引导训练 (CFG Training):
           - 80%时间: 模型看到标签 y (条件生成)
           - 20%时间: 标签置为-1 (无条件生成)
           - 生成时可通过guidance_scale控制类别特征强度
        
        总损失: L_total = L_diff + λ * L_recon
        
        Args:
            x_0: (B, D) - clean data
            y: (B,) - labels
            mask_prob: float - probability of masking dependent features (default=0.5)
            mask_lambda: float - weight for reconstruction loss (default=0.1)
            p_uncond: float - probability of unconditional training for CFG (default=0.2)
            
        Returns:
            total_loss: scalar
        """
        if int(self.scaler_fitted.item()) != 1:
            raise RuntimeError("TabDDPM scaler not fitted. Call fit_scaler(...) before training.")

        B = x_0.shape[0]
        device = x_0.device

        x_0_scaled = self.transform(x_0)
        
        # Sample timesteps
        t = torch.randint(0, self.timesteps, (B,), device=device)
        
        # Classifier-free guidance training: randomly drop condition
        # With probability p_uncond, replace label with null token (-1)
        y_train = y.clone()
        uncond_mask = torch.rand(B, device=device) < p_uncond
        y_train[uncond_mask] = -1  # Null token for unconditional training
        
        # Extract conditional and dependent features
        # Use advanced indexing to extract specific columns
        x_cond = x_0_scaled[:, self.cond_indices]  # (B, len(cond_indices))
        x_dep = x_0_scaled[:, self.dep_indices]    # (B, len(dep_indices))
        
        # Debug: verify dimensions
        if x_cond.shape[-1] != len(self.cond_indices):
            logger.error(f"x_cond dimension mismatch:")
            logger.error(f"  Expected: {len(self.cond_indices)}")
            logger.error(f"  Actual: {x_cond.shape[-1]}")
            logger.error(f"  x_0 shape: {x_0_scaled.shape}")
            logger.error(f"  cond_indices: {self.cond_indices}")
            raise ValueError(f"x_cond dimension mismatch: expected {len(self.cond_indices)}, got {x_cond.shape[-1]}")
        
        # Create mask for dependent features
        mask = torch.rand(B, len(self.dep_indices), device=device) > mask_prob
        x_dep_masked = x_dep * mask
        
        # Reconstruct full feature vector
        x_0_masked = x_0_scaled.clone()
        x_0_masked[:, self.dep_indices] = x_dep_masked
        
        # Sample noise
        noise = torch.randn_like(x_0_scaled)
        
        # Forward diffusion
        x_t = self.q_sample(x_0_masked, t, noise)
        
        # Predict noise (use y_train which may contain null tokens)
        predicted_noise = self.predict_noise(x_t, t, y_train, x_cond)
        
        # Diffusion loss
        diffusion_loss = F.mse_loss(predicted_noise, noise)
        
        # Reconstruction loss on masked dependent features
        recon_loss = F.mse_loss(predicted_noise[:, self.dep_indices] * (~mask), 
                                noise[:, self.dep_indices] * (~mask))
        
        # Total loss
        total_loss = diffusion_loss + mask_lambda * recon_loss
        
        return total_loss
    
    @torch.no_grad()
    def p_sample(self, x_t, t, y, x_cond, guidance_scale=1.0):
        """
        Reverse diffusion: denoise one step
        
        Args:
            x_t: (B, D) - noisy data at timestep t
            t: (B,) - timesteps
            y: (B,) - labels
            x_cond: (B, C) - conditional features
            guidance_scale: float - classifier-free guidance scale
            
        Returns:
            x_{t-1}: (B, D) - denoised data at timestep t-1
        """
        # Conditional noise prediction
        noise_cond = self.predict_noise(x_t, t, y, x_cond)
        
        # Unconditional noise prediction (for classifier-free guidance)
        if guidance_scale != 1.0:
            y_uncond = torch.ones_like(y) * -1  # Null token
            noise_uncond = self.predict_noise(x_t, t, y_uncond, x_cond)
            
            # Guided noise prediction
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = noise_cond
        
        # Compute x_{t-1}
        alpha_t = self.alphas[t].unsqueeze(-1)
        alpha_cumprod_t = self.alphas_cumprod[t].unsqueeze(-1)
        beta_t = self.betas[t].unsqueeze(-1)
        
        # Mean of p(x_{t-1} | x_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - alpha_cumprod_t)) * noise_pred)
        
        # Add noise (except for t=0)
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = beta_t
            x_prev = mean + torch.sqrt(variance) * noise
        else:
            x_prev = mean
        
        return x_prev
    
    @torch.no_grad()
    def sample(self, n_samples, y, x_cond=None, guidance_scale=1.0, device='cpu'):
        """
        Generate samples using DDIM sampling (逆向扩散过程)
        
        生成流程：
        1. 从纯噪声开始: x_T ~ N(0, I)
        2. 逐步去噪: x_T → x_{T-1} → ... → x_1 → x_0
        3. 每步使用分类器自由引导 (CFG):
           ε_guided = ε_uncond + w * (ε_cond - ε_uncond)
           - w=1.0: 标准生成
           - w>1.0: 强化类别特征 (恶意更恶意, 良性更良性)
           - w<1.0: 弱化类别特征
        4. 最后强制条件特征: x_0[cond_indices] = x_cond (硬约束)
        
        为什么用DDIM而不是DDPM？
        - DDPM: 需要1000步采样，慢
        - DDIM: 只需50步采样，快10-20倍，质量相当
        
        Args:
            n_samples: int - number of samples to generate
            y: (n_samples,) - labels (条件标签)
            x_cond: (n_samples, len(cond_indices)) - conditional features (Direction, ValidMask)
            guidance_scale: float - CFG引导强度 (1.5=恶意, 1.2=良性)
            device: torch device
            
        Returns:
            x_0: (n_samples, D) - generated samples (生成的干净样本)
        """
        if int(self.scaler_fitted.item()) != 1:
            raise RuntimeError("TabDDPM scaler not fitted. Call fit_scaler(...) before sampling.")

        cond_dim = len(self.cond_indices)
        if cond_dim > 0:
            if x_cond is None:
                raise ValueError("x_cond must be provided when cond_indices is non-empty")
            x_cond_raw = x_cond
            x_cond_scaled = (x_cond_raw - self.feature_mean[self.cond_indices]) / self.feature_std[self.cond_indices]
        else:
            x_cond_raw = None
            x_cond_scaled = torch.empty((n_samples, 0), device=device)

        x_t = torch.randn(n_samples, self.input_dim, device=device)

        sampling_steps = self.config.DDPM_SAMPLING_STEPS
        step_size = self.timesteps // sampling_steps

        for i in reversed(range(sampling_steps)):
            t = torch.full((n_samples,), i * step_size, device=device, dtype=torch.long)
            x_t = self.p_sample(x_t, t, y, x_cond_scaled, guidance_scale)

        x_0_scaled = x_t
        if cond_dim > 0:
            x_0_scaled[:, self.cond_indices] = x_cond_scaled

        x_0_raw = self.inverse_transform(x_0_scaled)
        if cond_dim > 0:
            x_0_raw[:, self.cond_indices] = x_cond_raw
        x_0_raw = self.project_protocol_constraints(x_0_raw)
        return x_0_raw

    def augment_feature_dataset(self, Z_clean, y_clean, correction_weight):
        device = next(self.parameters()).device

        Z_clean = np.asarray(Z_clean, dtype=np.float32)
        y_clean = np.asarray(y_clean).astype(int)
        correction_weight = np.asarray(correction_weight, dtype=np.float32)

        if Z_clean.ndim != 2 or Z_clean.shape[1] != self.input_dim:
            raise ValueError(f"Z_clean must have shape (N, {self.input_dim}), got {Z_clean.shape}")

        keep_mask = correction_weight > 0.0
        Z_keep = Z_clean[keep_mask]
        y_keep = y_clean[keep_mask]
        w_keep = correction_weight[keep_mask]

        logger.info(f"Augmenting feature dataset: {len(Z_keep)} clean samples retained")

        augmentation_mode = getattr(self.config, 'AUGMENTATION_MODE', 'weighted')
        
        # v2.5: 支持类别特定增强倍数
        if augmentation_mode == 'class_specific':
            benign_mult = int(getattr(self.config, 'STAGE2_BENIGN_AUG_MULTIPLIER', 3))
            malicious_mult = int(getattr(self.config, 'STAGE2_MALICIOUS_AUG_MULTIPLIER', 1))
            
            multipliers = np.zeros(len(w_keep), dtype=int)
            benign_mask = y_keep == 0
            malicious_mask = y_keep == 1
            multipliers[benign_mask] = benign_mult
            multipliers[malicious_mask] = malicious_mult
            
            logger.info(f"Using class-specific augmentation mode:")
            logger.info(f"  Benign: {benign_mult}x ({benign_mask.sum()} samples)")
            logger.info(f"  Malicious: {malicious_mult}x ({malicious_mask.sum()} samples)")
            
        elif augmentation_mode == 'fixed':
            fixed_multiplier = int(getattr(self.config, 'STAGE2_FEATURE_AUG_MULTIPLIER', 5))
            multipliers = np.full(len(w_keep), fixed_multiplier, dtype=int)
            logger.info(f"Using fixed augmentation mode: {fixed_multiplier}x for all samples")
        else:
            tier1_min_w = float(getattr(self.config, 'STAGE2_FEATURE_TIER1_MIN_WEIGHT', 0.9))
            tier1_mult = int(getattr(self.config, 'STAGE2_FEATURE_TIER1_MULTIPLIER', 10))
            tier2_min_w = float(getattr(self.config, 'STAGE2_FEATURE_TIER2_MIN_WEIGHT', 0.7))
            tier2_mult = int(getattr(self.config, 'STAGE2_FEATURE_TIER2_MULTIPLIER', 5))
            low_mult = int(getattr(self.config, 'STAGE2_FEATURE_LOWCONF_MULTIPLIER', 0))

            def compute_augmentation_multiplier(weight):
                if weight >= tier1_min_w:
                    return tier1_mult
                if weight >= tier2_min_w:
                    return tier2_mult
                return low_mult
            multipliers = np.array([compute_augmentation_multiplier(w) for w in w_keep], dtype=int)
            logger.info(f"Using weighted augmentation mode")

        unique_mults, counts = np.unique(multipliers, return_counts=True)
        logger.info("Feature augmentation multiplier distribution:")
        for mult, count in zip(unique_mults, counts):
            logger.info(f"  {mult}x: {count} samples ({100*count/len(multipliers):.1f}%)")

        Z_synthetic_list = []
        y_synthetic_list = []
        w_synthetic_list = []

        use_weighted_sampling = bool(getattr(self.config, 'AUGMENT_USE_WEIGHTED_SAMPLING', True))
        enable_quality_check = bool(getattr(self.config, 'AUGMENT_QUALITY_CHECK', True))

        for class_label in np.unique(y_keep):
            class_mask = y_keep == class_label
            Z_class = Z_keep[class_mask]
            mult_class = multipliers[class_mask]
            w_class = w_keep[class_mask]

            # v2.5: 提高模板质量要求
            min_w_template = float(getattr(self.config, 'AUGMENT_TEMPLATE_MIN_WEIGHT', 0.8))
            min_w_template_hard = float(getattr(self.config, 'AUGMENT_TEMPLATE_MIN_WEIGHT_HARD', 0.6))
            template_mask = w_class >= min_w_template
            if template_mask.sum() == 0:
                logger.warning(f"No templates meet quality threshold {min_w_template}, using hard threshold {min_w_template_hard}")
                template_mask = w_class >= min_w_template_hard
            if template_mask.sum() == 0:
                logger.warning(f"No templates meet hard threshold, using all samples")
                template_mask = np.ones_like(w_class, dtype=bool)

            Z_templates_pool = Z_class[template_mask]
            w_templates_pool = w_class[template_mask]
            mult_templates_pool = mult_class[template_mask]
            n_generate = int(mult_templates_pool.sum())
            if n_generate <= 0:
                continue

            logger.info(f"Class {class_label}: {len(Z_templates_pool)} templates (avg weight: {w_templates_pool.mean():.3f})")

            if use_weighted_sampling:
                p = w_templates_pool.astype(np.float64)
                p_sum = float(p.sum())
                if p_sum <= 0:
                    p = np.ones_like(p, dtype=np.float64) / max(len(p), 1)
                else:
                    p = p / p_sum
                idx = np.random.choice(len(Z_templates_pool), size=n_generate, replace=True, p=p)
                w_templates = w_templates_pool[idx]
            else:
                w_templates = np.repeat(w_templates_pool, repeats=mult_templates_pool, axis=0)
            n_generate = int(w_templates.shape[0])

            guidance_scale = self.config.GUIDANCE_MALICIOUS if int(class_label) == 1 else self.config.GUIDANCE_BENIGN
            logger.info(f"Generating {n_generate} synthetic feature samples for class {class_label} (guidance={guidance_scale})")

            y_tensor = torch.full((n_generate,), int(class_label), dtype=torch.long, device=device)
            Z_generated = []
            batch_size = 4096
            for i in range(0, n_generate, batch_size):
                end_i = min(i + batch_size, n_generate)
                batch_y = y_tensor[i:end_i]
                batch_generated = self.sample(
                    n_samples=len(batch_y),
                    y=batch_y,
                    x_cond=None,
                    guidance_scale=guidance_scale,
                    device=device
                )
                Z_generated.append(batch_generated.detach().cpu().numpy().astype(np.float32))
            Z_generated = np.concatenate(Z_generated, axis=0)
            
            # v2.5: 数据质量检查
            if enable_quality_check and len(Z_generated) > 0:
                Z_generated = self._quality_check_samples(Z_generated, Z_templates_pool, class_label)
            
            # v2.6: Mixup增强（在质量检查后应用）
            enable_mixup = bool(getattr(self.config, 'AUGMENT_MIXUP_ENABLED', False))
            if enable_mixup and len(Z_generated) > 0:
                # 根据类别选择Mixup强度
                if int(class_label) == 0:
                    mixup_alpha = float(getattr(self.config, 'AUGMENT_MIXUP_ALPHA_BENIGN', 0.1))
                else:
                    mixup_alpha = float(getattr(self.config, 'AUGMENT_MIXUP_ALPHA_MALICIOUS', 0.3))
                
                Z_generated = self._mixup_augmentation(Z_templates_pool, Z_generated, alpha=mixup_alpha)

            Z_synthetic_list.append(Z_generated)
            y_synthetic_list.append(np.full(len(Z_generated), int(class_label), dtype=int))
            w_synthetic_list.append(w_templates[:len(Z_generated)].astype(np.float32))

        if len(Z_synthetic_list) == 0:
            logger.info("No synthetic feature samples generated; returning original features")
            return Z_keep, y_keep, w_keep

        Z_synth = np.concatenate(Z_synthetic_list, axis=0)
        y_synth = np.concatenate(y_synthetic_list, axis=0)
        w_synth = np.concatenate(w_synthetic_list, axis=0)

        Z_aug = np.concatenate([Z_keep, Z_synth], axis=0)
        y_aug = np.concatenate([y_keep, y_synth], axis=0)
        w_aug = np.concatenate([w_keep, w_synth], axis=0)
        
        # 输出增强统计
        logger.info(f"Augmentation complete:")
        logger.info(f"  Original: {len(Z_keep)} samples")
        logger.info(f"  Synthetic: {len(Z_synth)} samples")
        logger.info(f"  Total: {len(Z_aug)} samples")
        logger.info(f"  Benign: {(y_aug==0).sum()} ({100*(y_aug==0).sum()/len(y_aug):.1f}%)")
        logger.info(f"  Malicious: {(y_aug==1).sum()} ({100*(y_aug==1).sum()/len(y_aug):.1f}%)")

        return Z_aug, y_aug, w_aug
    
    def _quality_check_samples(self, Z_generated, Z_templates, class_label):
        """
        v2.5: 质量检查生成的样本，过滤离群点
        """
        if len(Z_generated) == 0 or len(Z_templates) == 0:
            return Z_generated
        
        max_outlier_ratio = float(getattr(self.config, 'AUGMENT_MAX_OUTLIER_RATIO', 0.05))
        
        # 计算生成样本与模板的距离
        from scipy.spatial.distance import cdist
        distances = cdist(Z_generated, Z_templates, metric='euclidean')
        min_distances = distances.min(axis=1)
        
        # 使用IQR方法检测离群点
        q75, q25 = np.percentile(min_distances, [75, 25])
        iqr = q75 - q25
        outlier_threshold = q75 + 1.5 * iqr
        
        outlier_mask = min_distances > outlier_threshold
        n_outliers = outlier_mask.sum()
        outlier_ratio = n_outliers / len(Z_generated)
        
        if outlier_ratio > max_outlier_ratio:
            # 过滤离群点
            Z_filtered = Z_generated[~outlier_mask]
            logger.info(f"  Quality check (class {class_label}): removed {n_outliers}/{len(Z_generated)} outliers ({100*outlier_ratio:.1f}%)")
            return Z_filtered
        else:
            logger.info(f"  Quality check (class {class_label}): passed ({n_outliers} outliers, {100*outlier_ratio:.1f}%)")
            return Z_generated
    
    def _mixup_augmentation(self, Z_templates, Z_synthetic, alpha=0.2):
        """
        v2.6: Mixup增强 - 在模板和生成样本之间插值
        
        核心思想：
        - 生成样本可能过于集中在高密度区域
        - 通过与真实模板混合，增加样本多样性
        - 同时保持样本的真实性（不会偏离分布太远）
        
        公式：
        Z_mixed = λ * Z_synthetic + (1-λ) * Z_template
        其中 λ ~ Beta(α, α)
        
        Args:
            Z_templates: (N_template, D) - 真实模板样本
            Z_synthetic: (N_syn, D) - TabDDPM生成的样本
            alpha: Beta分布参数（控制混合强度）
                - alpha=0.1: 弱混合（生成样本占主导，90%权重）
                - alpha=0.3: 强混合（更多样性，70%权重）
                - alpha=1.0: 均匀混合（50%权重）
        
        Returns:
            Z_mixed: (N_syn, D) - 混合后的样本
        """
        if alpha <= 0 or len(Z_templates) == 0:
            return Z_synthetic
        
        n_samples = len(Z_synthetic)
        
        # 从Beta分布采样混合系数
        lambda_mix = np.random.beta(alpha, alpha, size=n_samples)
        
        # 随机选择模板样本进行混合
        idx = np.random.choice(len(Z_templates), size=n_samples, replace=True)
        Z_templates_sampled = Z_templates[idx]
        
        # 线性插值
        Z_mixed = lambda_mix[:, None] * Z_synthetic + (1 - lambda_mix[:, None]) * Z_templates_sampled
        
        logger.info(f"  Mixup augmentation: alpha={alpha:.2f}, avg_lambda={lambda_mix.mean():.3f}")
        
        return Z_mixed

    def augment_dataset(self, X_clean, y_clean, action_mask, correction_weight,
                        density_scores, cl_confidence, knn_confidence,
                        augmentation_ratio=4):
        """
        Augment dataset with generated samples (weight-based multiplier strategy)
        
        增强策略详解（v2 - 权重倍数增强）：
        
        1. 权重倍数映射 (Weight-Based Multiplier):
           - 权重 ≥ 0.9: 4倍增强 (Tier 1/2/3/5a/5b - 高质量样本)
           - 权重 ≥ 0.4: 2倍增强 (Tier 4 - 中等质量样本)
           - 权重 < 0.4: 不增强 (Dropped - 低质量样本)
        
        2. 公式:
           multiplier = compute_augmentation_multiplier(weight)
           
           def compute_augmentation_multiplier(weight):
               if weight >= 0.9:
                   return 4
               elif weight >= 0.4:
                   return 2
               else:
                   return 0
        
        3. 条件生成机制:
           - 从高权重样本中采样条件特征 (Direction, ValidMask)
           - 模型生成依赖特征 (Length, BurstSize)
           - 保证生成样本符合协议约束
        
        4. 类别平衡策略:
           - 每个类别独立生成
           - 恶意类用 guidance=1.5 (强化攻击特征)
           - 良性类用 guidance=1.2 (保持正常特征)
        
        5. 权重继承:
           - 原始样本: 保留 correction_weight
           - 合成样本: 继承模板样本的权重（而非固定为1.0）
           - 原理: 合成样本的质量取决于模板样本的质量
        
        为什么不会过拟合？
        - 每次生成都从随机噪声开始
        - 训练时每个样本都加了不同的噪声
        - 只用条件特征，依赖特征是学习分布生成的
        
        Args:
            X_clean: (N, L, D) - clean original samples (sequences)
            y_clean: (N,) - clean labels
            action_mask: (N,) - 0=keep, 1=flip, 2=drop, 3=reweight
            correction_weight: (N,) - lifecycle weights from Hybrid Court
            density_scores: (N,) - MADE density scores (raw)
            cl_confidence: (N,) - CL max probability per sample
            knn_confidence: (N,) - KNN consistency per sample
            augmentation_ratio: int - deprecated, not used in v2
            
        Returns:
            X_augmented: (N_aug, L, D) - augmented data
            y_augmented: (N_aug,) - augmented labels
            sample_weights: (N_aug,) - weights for training
        """
        raise RuntimeError(
            "augment_dataset(raw/sequence-level) is not supported in the fixed lite4 + dual-stream design. "
            "Use augment_feature_dataset(Z_clean, y_clean, weights_clean) instead."
        )
