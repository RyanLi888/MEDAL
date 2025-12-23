"""
ML投票标签矫正实验
==================

使用多个机器学习模型对低质量样本进行投票矫正，并生成详细的分析报告和可视化图表。

功能:
1. 基于质量分数将样本分为高质量和低质量两组
2. 使用高质量样本训练多个ML模型
3. 对低质量样本进行投票矫正
4. 生成详细的分析报告和可视化图表

输出:
- output/report.txt          - 详细分析报告
- output/figures/            - 可视化图表
- output/summary.json        - 实验摘要
"""
import argparse
import os
import sys
import json
import logging
import csv
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MoudleCode.utils.config import config
from MoudleCode.label_correction.hybrid_court import HybridCourt
from MoudleCode.label_correction.hybrid_court_experimental import HybridCourtExperimental
from MoudleCode.feature_extraction.backbone import MicroBiMambaBackbone
from MoudleCode.utils.helpers import inject_label_noise, calculate_metrics, set_seed

try:
    from preprocess import check_preprocessed_exists, load_preprocessed
    PREPROCESS_AVAILABLE = True
except ImportError:
    PREPROCESS_AVAILABLE = False

# 配置中文字体
def _configure_matplotlib_chinese_fonts():
    try:
        # 直接使用可用的中文字体
        mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Serif CJK JP', 'Noto Mono']
        mpl.rcParams['axes.unicode_minus'] = False
        # 清除字体缓存
        font_manager._rebuild()
    except Exception:
        pass

_configure_matplotlib_chinese_fonts()


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def _softmax_max_prob(pred_probs: np.ndarray) -> np.ndarray:
    if pred_probs.ndim != 2:
        raise ValueError(f"pred_probs must have shape (N, C), got {pred_probs.shape}")
    return np.max(pred_probs, axis=1)


def _minmax01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn + eps)).astype(np.float32)


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _build_models(random_state: int):
    """构建多个ML模型"""
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    models = {}
    models["GaussianNB"] = GaussianNB()
    models["AdaBoost"] = AdaBoostClassifier(random_state=random_state)
    models["LDA"] = LinearDiscriminantAnalysis()
    models["SVM"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, kernel="rbf", C=1.0, gamma="scale", random_state=random_state))
    ])
    models["RandomForest"] = RandomForestClassifier(n_estimators=400, random_state=random_state, n_jobs=-1)
    models["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=random_state))
    ])

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss",
            n_jobs=-1, random_state=random_state, verbosity=0
        )
    except Exception:
        models["XGBoost"] = GradientBoostingClassifier(random_state=random_state)

    return models


def _fit_estimator(model, X, y, sample_weight=None):
    if sample_weight is None:
        model.fit(X, y)
        return
    try:
        model.fit(X, y, sample_weight=sample_weight)
        return
    except ValueError:
        if isinstance(model, Pipeline) and len(model.steps) > 0:
            last_step_name = model.steps[-1][0]
            model.fit(X, y, **{f"{last_step_name}__sample_weight": sample_weight})
            return
        model.fit(X, y)
        return
    except TypeError:
        model.fit(X, y)


def _predict_proba_pos1(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1].astype(np.float32)
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = np.asarray(s).astype(np.float32)
        return (1.0 / (1.0 + np.exp(-s))).astype(np.float32)
    return model.predict(X).astype(np.float32)


def _predict_proba_matrix(fitted: dict, X: np.ndarray) -> tuple:
    names = list(fitted.keys())
    mats = []
    for name, model in fitted.items():
        mats.append(_predict_proba_pos1(model, X))
    mat = np.stack(mats, axis=1) if mats else np.zeros((len(X), 0), dtype=np.float32)
    return mat, names


def _stacked_predict_proba(models: dict, X_train: np.ndarray, y_train: np.ndarray, X_vote: np.ndarray, seed: int, sample_weight=None) -> np.ndarray:
    names = list(models.keys())
    n_models = len(names)
    oof = np.zeros((len(X_train), n_models), dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_va = X_train[va_idx]
        w_tr = sample_weight[tr_idx] if sample_weight is not None else None
        for mi, name in enumerate(names):
            est = clone(models[name])
            _fit_estimator(est, X_tr, y_tr, sample_weight=w_tr)
            oof[va_idx, mi] = _predict_proba_pos1(est, X_va)
    meta = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
    meta.fit(oof, y_train)
    base_vote = np.zeros((len(X_vote), n_models), dtype=np.float32)
    for mi, name in enumerate(names):
        est = clone(models[name])
        _fit_estimator(est, X_train, y_train, sample_weight=sample_weight)
        base_vote[:, mi] = _predict_proba_pos1(est, X_vote)
    return meta.predict_proba(base_vote)[:, 1].astype(np.float32)


def _mlp_predict_proba(X_train: np.ndarray, y_train: np.ndarray, X_vote: np.ndarray, seed: int, device, sample_weight=None,
                      hidden: int = 128, epochs: int = 200, lr: float = 1e-3, weight_decay: float = 1e-4,
                      batch_size: int = 64) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train.astype(np.float32), dtype=torch.float32, device=device).view(-1, 1)
    X_vote_t = torch.as_tensor(X_vote, dtype=torch.float32, device=device)
    if sample_weight is None:
        w_t = torch.ones((len(X_train_t), 1), dtype=torch.float32, device=device)
    else:
        w = np.asarray(sample_weight, dtype=np.float32).reshape(-1, 1)
        w_t = torch.as_tensor(w, dtype=torch.float32, device=device)

    d = X_train_t.shape[1]
    model = nn.Sequential(
        nn.Linear(d, hidden),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(hidden, 1),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    idx = torch.arange(len(X_train_t), device=device)
    for _ in range(int(epochs)):
        perm = idx[torch.randperm(len(idx))]
        for start in range(0, len(perm), batch_size):
            b = perm[start:start + batch_size]
            xb = X_train_t[b]
            yb = y_train_t[b]
            wb = w_t[b]
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss = (loss * wb).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_vote_t)
        prob = torch.sigmoid(logits).view(-1).detach().cpu().numpy().astype(np.float32)
    return prob


def _extract_backbone_features(backbone: MicroBiMambaBackbone, X: np.ndarray, device, logger, batch_size: int = 64) -> np.ndarray:
    backbone.to(device)
    backbone.freeze()
    backbone.eval()

    feats = []
    with torch.no_grad():
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        total_batches = (len(X_tensor) + batch_size - 1) // batch_size
        for bi, start in enumerate(range(0, len(X_tensor), batch_size), start=1):
            X_batch = X_tensor[start:start + batch_size]
            z = backbone(X_batch, return_sequence=False)
            feats.append(z.detach().cpu().numpy())
            if bi % 10 == 0 or bi == total_batches:
                logger.info(f"  特征提取进度: {bi}/{total_batches} batches ({bi/total_batches*100:.1f}%)")

    return np.concatenate(feats, axis=0) if feats else np.empty((0, 0), dtype=np.float32)


def _run_from_preprocessed(args, logger):
    if not PREPROCESS_AVAILABLE:
        logger.error("❌ 无法导入 preprocess.py（无法加载预处理数据）。请确保从项目根目录运行。")
        return

    set_seed(args.seed)

    reuse_cache_dir = str(getattr(args, 'reuse_cache_dir', '') or '')
    save_cache_dir = str(getattr(args, 'save_cache_dir', '') or '')

    if reuse_cache_dir:
        cache_dir = reuse_cache_dir
        logger.info("步骤1: 复用缓存（跳过特征提取与HybridCourt）")
        logger.info("-"*70)
        required = [
            "features.npy",
            "y_true.npy",
            "y_noisy.npy",
            "y_corrected_hc.npy",
            "action_mask.npy",
            "correction_weight.npy",
        ]
        missing = [fn for fn in required if not os.path.exists(os.path.join(cache_dir, fn))]
        if missing:
            logger.error(f"❌ reuse_cache_dir 缺少文件: {missing} (dir={cache_dir})")
            return

        features = np.load(os.path.join(cache_dir, "features.npy"))
        y_true = np.load(os.path.join(cache_dir, "y_true.npy")).astype(int)
        y_noisy = np.load(os.path.join(cache_dir, "y_noisy.npy")).astype(int)
        y_corrected = np.load(os.path.join(cache_dir, "y_corrected_hc.npy")).astype(int)
        action_mask = np.load(os.path.join(cache_dir, "action_mask.npy")).astype(int)
        correction_weight = np.load(os.path.join(cache_dir, "correction_weight.npy")).astype(np.float32)
        noise_mask = (y_noisy != y_true)
        features_source = os.path.join(cache_dir, "features.npy")
    else:
        if not check_preprocessed_exists(args.data_split):
            logger.error(f"❌ 预处理文件不存在: output/preprocessed/{args.data_split}_X.npy 等")
            return

        X_seq, y_true, _ = load_preprocessed(args.data_split)
        y_true = y_true.astype(int)

        if args.noise_rate > 0:
            y_noisy, noise_mask = inject_label_noise(y_true, args.noise_rate, num_classes=2)
        else:
            y_noisy = y_true.copy()
            noise_mask = np.zeros(len(y_true), dtype=bool)

        if args.features_npy:
            if not os.path.exists(args.features_npy):
                logger.error(f"❌ features_npy 不存在: {args.features_npy}")
                return
            features = np.load(args.features_npy)
            features_source = args.features_npy
        else:
            backbone_path = args.backbone or os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
            if not os.path.exists(backbone_path):
                logger.error(f"❌ backbone 权重不存在: {backbone_path}")
                return
            backbone = MicroBiMambaBackbone(config)
            backbone.load_state_dict(torch.load(backbone_path, map_location=config.DEVICE))
            features_source = backbone_path
            logger.info("步骤2: 特征提取")
            logger.info("-"*70)
            features = _extract_backbone_features(backbone, X_seq, config.DEVICE, logger, batch_size=64)

    if len(features) != len(y_true):
        logger.error(f"❌ 特征数量与标签数量不一致: features={len(features)}, labels={len(y_true)}")
        return

    if not reuse_cache_dir:
        logger.info("步骤3: HybridCourt 标签矫正")
        logger.info("-"*70)
        if getattr(args, 'hc_impl', 'experimental') == 'original':
            hybrid_court = HybridCourt(config)
        else:
            hybrid_court = HybridCourtExperimental(config)
        y_corrected, action_mask, _, correction_weight, _, _, _ = hybrid_court.correct_labels(
            features, y_noisy, device=config.DEVICE
        )

        if save_cache_dir:
            os.makedirs(save_cache_dir, exist_ok=True)
            np.save(os.path.join(save_cache_dir, "features.npy"), features)
            np.save(os.path.join(save_cache_dir, "y_true.npy"), y_true)
            np.save(os.path.join(save_cache_dir, "y_noisy.npy"), y_noisy)
            np.save(os.path.join(save_cache_dir, "y_corrected_hc.npy"), y_corrected)
            np.save(os.path.join(save_cache_dir, "action_mask.npy"), action_mask.astype(np.int32))
            np.save(os.path.join(save_cache_dir, "correction_weight.npy"), correction_weight.astype(np.float32))
            meta = {
                "seed": int(args.seed),
                "noise_rate": float(args.noise_rate),
                "data_split": str(args.data_split),
                "hc_impl": str(getattr(args, 'hc_impl', 'experimental')),
                "features_source": str(features_source),
            }
            with open(os.path.join(save_cache_dir, "cache_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.train_on == "keep":
        keep_mask = action_mask == 0
    elif args.train_on == "keep_flip":
        keep_mask = (action_mask == 0) | (action_mask == 1)
    else:
        keep_mask = action_mask != 2

    # 默认仅对 Reweight(action=3) 做投票矫正；可选把 Drop(action=2) 也加入（高置信覆盖）
    if getattr(args, 'vote_include_drop', False):
        vote_mask = ((action_mask == 3) | (action_mask == 2)) & (~keep_mask)
    else:
        vote_mask = (action_mask == 3) & (~keep_mask)

    idx_keep = np.where(keep_mask)[0]
    idx_vote = np.where(vote_mask)[0]
    idx_rest = np.where(~keep_mask)[0]

    logger.info("步骤4: 划分Keep样本与待预测样本")
    logger.info("-"*70)
    logger.info(f"  总样本数: {len(y_true)}")
    logger.info(f"  真实标签分布: 正常={(y_true==0).sum()} 恶意={(y_true==1).sum()}")
    logger.info(f"  注入噪声样本: {int(noise_mask.sum())}/{len(y_true)} (rate={args.noise_rate:.2f})")
    logger.info(f"  Keep样本(训练): {len(idx_keep)} ({100*len(idx_keep)/len(y_true):.1f}%)")
    logger.info(f"  其余样本(非训练): {len(idx_rest)} ({100*len(idx_rest)/len(y_true):.1f}%)")
    logger.info(f"  投票矫正样本(Reweight): {len(idx_vote)} ({100*len(idx_vote)/len(y_true):.1f}%)")
    logger.info("")
    
    if len(idx_keep) == 0:
        logger.error("❌ Keep样本为空，无法训练模型。请尝试 --train_on nondrop")
        return
    
    rng = np.random.default_rng(args.seed)
    X_train = features[idx_keep]
    y_train = y_corrected[idx_keep].astype(int)
    train_sample_weight = correction_weight[idx_keep].astype(float)

    if args.train_on == "keep_flip":
        flip_in_train = (action_mask[idx_keep] == 1)
        if flip_in_train.any():
            train_sample_weight = train_sample_weight.copy()
            train_sample_weight[flip_in_train] *= float(getattr(args, "flip_train_weight", 0.5))

    if args.balance_train:
        idx0 = idx_keep[y_train == 0]
        idx1 = idx_keep[y_train == 1]
        n = min(len(idx0), len(idx1))
        if n > 0:
            idx0_s = rng.choice(idx0, size=n, replace=False)
            idx1_s = rng.choice(idx1, size=n, replace=False)
            idx_bal = np.concatenate([idx0_s, idx1_s])
            rng.shuffle(idx_bal)
            X_train = features[idx_bal]
            y_train = y_corrected[idx_bal].astype(int)
            train_sample_weight = correction_weight[idx_bal].astype(float)

            if args.train_on == "keep_flip":
                flip_in_train = (action_mask[idx_bal] == 1)
                if flip_in_train.any():
                    train_sample_weight = train_sample_weight.copy()
                    train_sample_weight[flip_in_train] *= float(getattr(args, "flip_train_weight", 0.5))

    logger.info("步骤5: 训练ML模型")
    logger.info("-"*70)
    
    models = _build_models(args.seed)
    logger.info(f"  使用 {len(models)} 个模型: {list(models.keys())}")
    logger.info("")
    fitted, accuracies = _fit_models(models, X_train, y_train, logger, sample_weight=train_sample_weight)
    logger.info("")
    
    logger.info("步骤6: 对其余样本进行投票预测")
    logger.info("-"*70)
    X_vote = features[idx_vote]
    model_names = list(fitted.keys())
    if len(idx_vote) == 0:
        vote_mat = np.zeros((0, len(model_names)), dtype=int)
        vote_sum = np.zeros((0,), dtype=int)
        score = np.zeros((0,), dtype=np.float32)
        y_vote_hard = np.zeros((0,), dtype=int)
        vote_rule = getattr(args, "vote_rule", "hard")
        band_low = int(getattr(args, "band_low", -1))
        band_high = int(getattr(args, "band_high", len(model_names) + 1))
        vote_score = getattr(args, "vote_score", "count")
        prob_threshold = float(getattr(args, "prob_threshold", 0.5))
        prob_low = float(getattr(args, "prob_low", 0.3))
        prob_high = float(getattr(args, "prob_high", 0.7))
        vote_apply_local = np.zeros((0,), dtype=bool)
        y_vote_final = np.zeros((0,), dtype=int)
        vote_distribution = [0] * (len(model_names) + 1)
        model_malicious_ratio = {n: 0.0 for n in model_names}

        y_pred_all = y_corrected.copy()
        metrics_vote = None
        report_vote = ""
    else:
        vote_mat, _ = _vote_predict(fitted, X_vote)
        vote_sum = vote_mat.sum(axis=1) if vote_mat.size else np.zeros((len(X_vote),), dtype=int)

        vote_score = getattr(args, "vote_score", "count")
        prob_threshold = float(getattr(args, "prob_threshold", 0.5))
        prob_low = float(getattr(args, "prob_low", 0.3))
        prob_high = float(getattr(args, "prob_high", 0.7))

        mlp_hidden = int(getattr(args, "mlp_hidden", 128))
        mlp_epochs = int(getattr(args, "mlp_epochs", 200))
        mlp_lr = float(getattr(args, "mlp_lr", 1e-3))
        mlp_weight_decay = float(getattr(args, "mlp_weight_decay", 1e-4))

        proba_mat, _ = _predict_proba_matrix(fitted, X_vote)
        mean_proba = proba_mat.mean(axis=1).astype(np.float32) if proba_mat.size else np.zeros((len(X_vote),), dtype=np.float32)
        stacked_proba = None
        if vote_score == "stacked":
            stacked_proba = _stacked_predict_proba(models, X_train, y_train, X_vote, seed=args.seed, sample_weight=train_sample_weight)
        mlp_proba = None
        if vote_score == "mlp":
            mlp_proba = _mlp_predict_proba(
                X_train, y_train, X_vote, seed=args.seed, device=config.DEVICE, sample_weight=train_sample_weight,
                hidden=mlp_hidden, epochs=mlp_epochs, lr=mlp_lr, weight_decay=mlp_weight_decay
            )

        if vote_score == "mean_proba":
            score = mean_proba
        elif vote_score == "stacked":
            score = stacked_proba
        elif vote_score == "mlp":
            score = mlp_proba
        else:
            score = vote_sum.astype(np.float32)

        if vote_score in ["mean_proba", "stacked", "mlp"]:
            y_vote_hard = (score >= prob_threshold).astype(int)
        else:
            y_vote_hard = (vote_sum >= args.vote_k).astype(int)

        # vote_rule:
        # - hard:  对所有 idx_vote 都用硬阈值覆盖
        # - band:  仅在低票/高票时覆盖，其余保留HC（避免把不确定样本改坏）
        # - none:  不覆盖（等价于只用HC结果）
        vote_rule = getattr(args, "vote_rule", "hard")
        band_low = int(getattr(args, "band_low", -1))
        band_high = int(getattr(args, "band_high", len(model_names) + 1))

        if vote_rule == "none":
            vote_apply_local = np.zeros((len(idx_vote),), dtype=bool)
            y_vote_final = y_corrected[idx_vote].astype(int)
        elif vote_rule == "band":
            if vote_score in ["mean_proba", "stacked", "mlp"]:
                vote_apply_local = (score <= prob_low) | (score >= prob_high)
                y_vote_final = y_corrected[idx_vote].astype(int).copy()
                y_vote_final[score <= prob_low] = 0
                y_vote_final[score >= prob_high] = 1
            else:
                vote_apply_local = (vote_sum <= band_low) | (vote_sum >= band_high)
                y_vote_final = y_corrected[idx_vote].astype(int).copy()
                y_vote_final[vote_sum <= band_low] = 0
                y_vote_final[vote_sum >= band_high] = 1
        else:
            vote_apply_local = np.ones((len(idx_vote),), dtype=bool)
            y_vote_final = y_vote_hard

        vote_distribution = [0] * (len(model_names) + 1)
        for v in vote_sum:
            vote_distribution[int(v)] += 1
        
        model_malicious_ratio = {}
        for i, name in enumerate(model_names):
            model_malicious_ratio[name] = float(vote_mat[:, i].mean()) if vote_mat.size else 0.0
        
        # y_final: 以 HybridCourt 的 y_corrected 为底座，只在 vote_rule 指定的子集做覆盖
        y_pred_all = y_corrected.copy()
        y_pred_all[idx_vote[vote_apply_local]] = y_vote_final[vote_apply_local]

        metrics_vote = calculate_metrics(y_true[idx_vote], y_pred_all[idx_vote]) if len(idx_vote) > 0 else None
        report_vote = classification_report(y_true[idx_vote], y_pred_all[idx_vote], digits=4, zero_division=0) if len(idx_vote) > 0 else ""

    metrics_all = calculate_metrics(y_true, y_pred_all)
    correction_accuracy = float(metrics_all.get("accuracy", 0.0))

    # baseline 对比：noisy / HC / final
    acc_noisy = float(np.mean(y_noisy == y_true))
    acc_hc = float(np.mean(y_corrected == y_true))

    sample_report_path = os.path.join(args.output_dir, "sample_analysis.csv")
    vote_sum_all = np.full((len(y_true),), -1, dtype=np.int32)
    vote_pred_all = np.full((len(y_true),), -1, dtype=np.int32)
    vote_sum_all[idx_vote] = vote_sum.astype(np.int32)
    vote_pred_all[idx_vote] = y_vote_final.astype(np.int32)

    vote_score_value_all = np.full((len(y_true),), -1.0, dtype=np.float32)
    if len(idx_vote) > 0:
        vote_score_value_all[idx_vote] = score.astype(np.float32)

    vote_applied_all = np.zeros((len(y_true),), dtype=np.uint8)
    if len(idx_vote) > 0:
        vote_applied_all[idx_vote[vote_apply_local]] = 1

    model_vote_all = {}
    for mi, name in enumerate(model_names):
        arr = np.full((len(y_true),), -1, dtype=np.int32)
        if vote_mat.size:
            arr[idx_vote] = vote_mat[:, mi].astype(np.int32)
        model_vote_all[name] = arr

    with open(sample_report_path, "w", encoding="utf-8", newline="") as fcsv:
        fieldnames = [
            "index",
            "y_true",
            "y_noisy",
            "y_corrected_hc",
            "y_final",
            "noise_injected",
            "action_mask",
            "correction_weight",
            "keep_mask",
            "vote_applied",
            "vote_sum",
            "vote_pred",
            "vote_score_value",
        ] + [f"vote_{n}" for n in model_names]
        wcsv = csv.DictWriter(fcsv, fieldnames=fieldnames)
        wcsv.writeheader()
        for i in range(len(y_true)):
            row = {
                "index": int(i),
                "y_true": int(y_true[i]),
                "y_noisy": int(y_noisy[i]),
                "y_corrected_hc": int(y_corrected[i]),
                "y_final": int(y_pred_all[i]),
                "noise_injected": int(bool(noise_mask[i])),
                "action_mask": int(action_mask[i]),
                "correction_weight": float(correction_weight[i]),
                "keep_mask": int(bool(keep_mask[i])),
                "vote_applied": int(vote_applied_all[i]),
                "vote_sum": int(vote_sum_all[i]),
                "vote_pred": int(vote_pred_all[i]),
                "vote_score_value": float(vote_score_value_all[i]),
            }
            for n in model_names:
                row[f"vote_{n}"] = int(model_vote_all[n][i])
            wcsv.writerow(row)
    
    results = {
        "mode": "from_preprocessed",
        "data_split": str(args.data_split),
        "noise_rate": float(args.noise_rate),
        "train_on": str(args.train_on),
        "features_source": features_source,
        "n_total": int(len(y_true)),
        "n_keep": int(len(idx_keep)),
        "n_rest": int(len(idx_rest)),
        "n_vote": int(len(idx_vote)),
        "n_vote_applied": int(vote_applied_all.sum()),
        "vote_rule": str(vote_rule),
        "band_low": int(band_low),
        "band_high": int(band_high),
        "vote_score": str(vote_score),
        "prob_threshold": float(prob_threshold),
        "prob_low": float(prob_low),
        "prob_high": float(prob_high),
        "true_benign": int((y_true == 0).sum()),
        "true_malicious": int((y_true == 1).sum()),
        "noise_injected": int(noise_mask.sum()),
        "action_keep": int((action_mask == 0).sum()),
        "action_flip": int((action_mask == 1).sum()),
        "action_drop": int((action_mask == 2).sum()),
        "action_reweight": int((action_mask == 3).sum()),
        "correction_weight": correction_weight,
        "train_samples": int(len(X_train)),
        "models": list(fitted.keys()),
        "model_accuracies": accuracies,
        "vote_distribution": vote_distribution,
        "model_malicious_ratio": model_malicious_ratio,
        "metrics_vote": metrics_vote,
        "metrics_all": metrics_all,
        "correction_accuracy": correction_accuracy,
        "report_vote": report_vote,
        "acc_noisy": acc_noisy,
        "acc_hc": acc_hc,
        "sample_report": sample_report_path,
    }
    
    if not getattr(args, 'skip_plots', False):
        fig_dir = os.path.join(args.output_dir, "figures")
        plot_quality_distribution(
            correction_weight, y_true, idx_keep, idx_rest,
            os.path.join(fig_dir, "1_质量分数分布.png"), logger
        )
        
        plot_vote_analysis(
            vote_mat, vote_sum, y_pred_all[idx_vote].astype(int), idx_vote, y_corrected, model_names,
            os.path.join(fig_dir, "2_投票分析.png"), logger
        )
        
        plot_feature_space(
            features, y_true, y_pred_all, idx_keep, idx_rest,
            os.path.join(fig_dir, "3_特征空间可视化.png"), logger
        )
    logger.info("")
    
    logger.info("步骤7: 生成分析报告")
    logger.info("-"*70)
    generate_report(args, results, os.path.join(args.output_dir, "report.txt"), logger)
    logger.info("")
    
    logger.info("步骤8: 保存数据文件")
    logger.info("-"*70)
    summary = {
        "mode": "from_preprocessed",
        "data_split": str(args.data_split),
        "noise_rate": float(args.noise_rate),
        "train_on": str(args.train_on),
        "n_total": int(len(y_true)),
        "n_keep": int(len(idx_keep)),
        "n_rest": int(len(idx_rest)),
        "n_vote": int(len(idx_vote)),
        "n_vote_applied": int(vote_applied_all.sum()),
        "vote_rule": str(vote_rule),
        "band_low": int(band_low),
        "band_high": int(band_high),
        "vote_score": str(vote_score),
        "prob_threshold": float(prob_threshold),
        "prob_low": float(prob_low),
        "prob_high": float(prob_high),
        "vote_k": int(args.vote_k),
        "eval_all": _to_jsonable(metrics_all),
        "eval_vote": _to_jsonable(metrics_vote),
        "correction_accuracy": correction_accuracy,
        "acc_noisy": acc_noisy,
        "acc_hc": acc_hc,
        "sample_report": sample_report_path,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("  ✓ summary.json")
    
    np.save(os.path.join(args.output_dir, "y_true.npy"), y_true)
    np.save(os.path.join(args.output_dir, "y_noisy.npy"), y_noisy)
    np.save(os.path.join(args.output_dir, "y_corrected_hc.npy"), y_corrected)
    np.save(os.path.join(args.output_dir, "action_mask.npy"), action_mask.astype(np.int32))
    np.save(os.path.join(args.output_dir, "y_pred_all.npy"), y_pred_all)
    np.save(os.path.join(args.output_dir, "features.npy"), features.astype(np.float32))
    np.save(os.path.join(args.output_dir, "idx_keep.npy"), idx_keep.astype(np.int64))
    np.save(os.path.join(args.output_dir, "idx_rest.npy"), idx_rest.astype(np.int64))
    np.save(os.path.join(args.output_dir, "keep_mask.npy"), keep_mask.astype(np.uint8))
    np.save(os.path.join(args.output_dir, "correction_weight.npy"), correction_weight)
    logger.info("  ✓ y_true.npy, y_noisy.npy, y_corrected_hc.npy, y_pred_all.npy")
    logger.info("  ✓ idx_keep.npy, idx_rest.npy, keep_mask.npy, correction_weight.npy")
    logger.info("")
    
    logger.info("="*70)
    logger.info("🎉 实验完成! (from_preprocessed)")
    logger.info("="*70)
    logger.info("")
    logger.info("📊 结果摘要:")
    logger.info(f"  baseline(noisy) acc={acc_noisy:.4f} | HC acc={acc_hc:.4f} | final acc={correction_accuracy:.4f}")
    logger.info(f"  eval_all:  acc={metrics_all['accuracy']:.4f}, f1(pos=1)={metrics_all['f1_pos']:.4f}")
    logger.info(f"  Correction accuracy: {correction_accuracy:.4f}")
    if metrics_vote is not None:
        logger.info(f"  eval_vote: acc={metrics_vote['accuracy']:.4f}, f1(pos=1)={metrics_vote['f1_pos']:.4f}")
    logger.info(f"  输出目录: {args.output_dir}")
    logger.info(f"  样本分析CSV: {sample_report_path}")


def _fit_models(models: dict, X_train: np.ndarray, y_train: np.ndarray, logger, sample_weight=None):
    """训练所有模型"""
    fitted = {}
    accuracies = {}
    
    for name, model in models.items():
        logger.info(f"  训练 {name}...")
        _fit_estimator(model, X_train, y_train, sample_weight=sample_weight)
        fitted[name] = model
        
        # 计算训练集准确率
        train_pred = model.predict(X_train)
        acc = (train_pred == y_train).mean()
        accuracies[name] = acc
        logger.info(f"    ✓ {name} 训练完成，训练集准确率: {acc*100:.2f}%")
    
    return fitted, accuracies


def _vote_predict(fitted: dict, X: np.ndarray) -> tuple:
    """对样本进行投票预测"""
    votes = []
    names = list(fitted.keys())
    
    for name, model in fitted.items():
        pred = model.predict(X)
        votes.append(pred.astype(int))
    
    vote_mat = np.stack(votes, axis=1) if votes else np.zeros((len(X), 0), dtype=int)
    return vote_mat, names


def plot_quality_distribution(quality, y_base, idx_high, idx_low, save_path, logger):
    """绘制质量分数分布图"""
    logger.info("绘制质量分数分布图...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 整体质量分数分布
    ax = axes[0]
    ax.hist(quality[y_base == 0], bins=50, alpha=0.6, label='正常流量', color='#2E86AB')
    ax.hist(quality[y_base == 1], bins=50, alpha=0.6, label='恶意流量', color='#A23B72')
    ax.axvline(np.percentile(quality, 50), color='red', linestyle='--', label='中位数')
    ax.set_xlabel('质量分数')
    ax.set_ylabel('样本数量')
    ax.set_title('样本质量分数分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 高/低质量样本分布
    ax = axes[1]
    high_q = quality[idx_high]
    low_q = quality[idx_low]
    ax.hist(high_q, bins=30, alpha=0.6, label=f'高质量 (n={len(idx_high)})', color='#06A77D')
    ax.hist(low_q, bins=30, alpha=0.6, label=f'低质量 (n={len(idx_low)})', color='#D62828')
    ax.set_xlabel('质量分数')
    ax.set_ylabel('样本数量')
    ax.set_title('高/低质量样本分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 质量分数箱线图
    ax = axes[2]
    data = [quality[y_base == 0], quality[y_base == 1]]
    bp = ax.boxplot(data, labels=['正常流量', '恶意流量'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2E86AB')
    bp['boxes'][1].set_facecolor('#A23B72')
    ax.set_ylabel('质量分数')
    ax.set_title('各类别质量分数箱线图')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ 已保存: {save_path}")


def plot_vote_analysis(vote_mat, vote_sum, y_low_pred, idx_low, y_base, model_names, save_path, logger):
    """绘制投票分析图"""
    logger.info("绘制投票分析图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 投票分布直方图
    ax = axes[0, 0]
    ax.hist(vote_sum, bins=range(len(model_names)+2), alpha=0.7, color='#2E86AB', edgecolor='black')
    ax.set_xlabel('投票数（判为恶意的模型数）')
    ax.set_ylabel('样本数量')
    ax.set_title('低质量样本投票分布')
    ax.set_xticks(range(len(model_names)+1))
    ax.grid(True, alpha=0.3)
    
    # 2. 各模型预测分布
    ax = axes[0, 1]
    model_preds = vote_mat.sum(axis=0) / len(vote_mat) * 100
    bars = ax.bar(model_names, model_preds, color='#06A77D', edgecolor='black')
    ax.set_xlabel('模型')
    ax.set_ylabel('预测为恶意的比例 (%)')
    ax.set_title('各模型对低质量样本的预测')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    for bar, val in zip(bars, model_preds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', 
                ha='center', va='bottom', fontsize=9)
    
    # 3. 模型一致性热力图
    ax = axes[1, 0]
    agreement = np.corrcoef(vote_mat.T)
    sns.heatmap(agreement, annot=True, fmt='.2f', cmap='RdYlGn', 
                xticklabels=model_names, yticklabels=model_names, ax=ax)
    ax.set_title('模型预测一致性 (相关系数)')
    
    # 4. 矫正前后对比
    ax = axes[1, 1]
    original_labels = y_base[idx_low]
    categories = ['保持正常', '正常→恶意', '恶意→正常', '保持恶意']
    counts = [
        ((original_labels == 0) & (y_low_pred == 0)).sum(),
        ((original_labels == 0) & (y_low_pred == 1)).sum(),
        ((original_labels == 1) & (y_low_pred == 0)).sum(),
        ((original_labels == 1) & (y_low_pred == 1)).sum()
    ]
    colors = ['#2E86AB', '#F77F00', '#D62828', '#A23B72']
    bars = ax.bar(categories, counts, color=colors, edgecolor='black')
    ax.set_ylabel('样本数量')
    ax.set_title('低质量样本矫正结果')
    ax.set_xticklabels(categories, rotation=15, ha='right')
    ax.grid(True, alpha=0.3)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val), 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ 已保存: {save_path}")


def plot_feature_space(features, y_base, y_final, idx_high, idx_low, save_path, logger):
    """绘制特征空间可视化"""
    logger.info("绘制特征空间可视化 (t-SNE)...")
    logger.info("  正在进行t-SNE降维，请稍候...")
    
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 原始标签分布
    ax = axes[0]
    ax.scatter(features_2d[y_base == 0, 0], features_2d[y_base == 0, 1],
               c='#2E86AB', label='正常流量', alpha=0.5, s=20)
    ax.scatter(features_2d[y_base == 1, 0], features_2d[y_base == 1, 1],
               c='#A23B72', label='恶意流量', alpha=0.5, s=20)
    ax.set_xlabel('t-SNE 维度1')
    ax.set_ylabel('t-SNE 维度2')
    ax.set_title('原始标签分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 高/低质量样本分布
    ax = axes[1]
    ax.scatter(features_2d[idx_high, 0], features_2d[idx_high, 1],
               c='#06A77D', label=f'高质量 (n={len(idx_high)})', alpha=0.5, s=20)
    ax.scatter(features_2d[idx_low, 0], features_2d[idx_low, 1],
               c='#D62828', label=f'低质量 (n={len(idx_low)})', alpha=0.5, s=20)
    ax.set_xlabel('t-SNE 维度1')
    ax.set_ylabel('t-SNE 维度2')
    ax.set_title('高/低质量样本分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 矫正后标签分布
    ax = axes[2]
    ax.scatter(features_2d[y_final == 0, 0], features_2d[y_final == 0, 1],
               c='#2E86AB', label='正常流量', alpha=0.5, s=20)
    ax.scatter(features_2d[y_final == 1, 0], features_2d[y_final == 1, 1],
               c='#A23B72', label='恶意流量', alpha=0.5, s=20)
    # 标记被矫正的样本
    changed_mask = y_base != y_final
    if changed_mask.sum() > 0:
        ax.scatter(features_2d[changed_mask, 0], features_2d[changed_mask, 1],
                   facecolors='none', edgecolors='lime', s=100, linewidths=2,
                   label=f'已矫正 (n={changed_mask.sum()})')
    ax.set_xlabel('t-SNE 维度1')
    ax.set_ylabel('t-SNE 维度2')
    ax.set_title('ML投票矫正后标签分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ 已保存: {save_path}")


def generate_report(args, results, save_path, logger):
    """生成详细分析报告"""
    logger.info("生成分析报告...")
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ML投票标签矫正实验 - 详细分析报告\n")
        f.write("="*80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")

        if getattr(args, 'mode', 'legacy') == 'from_preprocessed':
            f.write("-"*80 + "\n")
            f.write("1. 实验配置\n")
            f.write("-"*80 + "\n")
            f.write(f"  模式:           {results.get('mode', 'from_preprocessed')}\n")
            f.write(f"  数据切分:       {results.get('data_split', '')}\n")
            f.write(f"  注入噪声率:     {results.get('noise_rate', 0.0):.2f}\n")
            f.write(f"  训练集口径:     {results.get('train_on', 'keep')}\n")
            f.write(f"  特征来源:       {results.get('features_source', '')}\n")
            f.write(f"  投票阈值:       {args.vote_k} (至少{args.vote_k}个模型同意才判为恶意)\n")
            f.write(f"  类别平衡训练:   {'是' if args.balance_train else '否'}\n")
            f.write(f"  随机种子:       {args.seed}\n")
            f.write("\n")

            f.write("-"*80 + "\n")
            f.write("2. 数据统计\n")
            f.write("-"*80 + "\n")
            f.write(f"  总样本数:       {results['n_total']}\n")
            f.write(f"  Keep样本(训练): {results['n_keep']} ({100*results['n_keep']/results['n_total']:.1f}%)\n")
            f.write(f"  其余样本(非训练): {results['n_rest']} ({100*results['n_rest']/results['n_total']:.1f}%)\n")
            f.write(f"  投票矫正样本(Reweight): {results.get('n_vote', 0)} ({100*results.get('n_vote', 0)/results['n_total']:.1f}%)\n")
            if 'n_vote_applied' in results:
                f.write(f"  实际覆盖样本:   {results.get('n_vote_applied', 0)}\n")
            if 'vote_rule' in results:
                f.write(f"  投票规则:       {results.get('vote_rule', '')} (band_low={results.get('band_low', '')}, band_high={results.get('band_high', '')})\n")
            if 'vote_score' in results:
                f.write(f"  投票打分:       {results.get('vote_score', '')} (prob_th={results.get('prob_threshold', '')}, prob_low={results.get('prob_low', '')}, prob_high={results.get('prob_high', '')})\n")
            f.write(f"  真实标签分布:   正常={results.get('true_benign', 0)}, 恶意={results.get('true_malicious', 0)}\n")
            f.write(f"  注入噪声样本:   {results.get('noise_injected', 0)}\n")
            f.write("\n")

            f.write("-"*80 + "\n")
            f.write("3. HybridCourt 动作统计\n")
            f.write("-"*80 + "\n")
            f.write(f"  Keep:           {results.get('action_keep', 0)}\n")
            f.write(f"  Flip:           {results.get('action_flip', 0)}\n")
            f.write(f"  Drop:           {results.get('action_drop', 0)}\n")
            f.write(f"  Reweight:       {results.get('action_reweight', 0)}\n")
            f.write("\n")

            f.write("-"*80 + "\n")
            f.write("4. 模型训练结果\n")
            f.write("-"*80 + "\n")
            f.write(f"  训练样本数:     {results.get('train_samples', 0)}\n")
            f.write(f"  使用的模型:     {len(results.get('models', []))} 个\n")
            f.write("\n")
            f.write(f"  {'模型名称':<20} {'训练集准确率':>15}\n")
            f.write(f"  {'-'*20} {'-'*15}\n")
            for name, acc in results.get('model_accuracies', {}).items():
                f.write(f"  {name:<20} {acc*100:>14.2f}%\n")
            f.write("\n")

            f.write("-"*80 + "\n")
            f.write("5. 投票预测结果\n")
            f.write("-"*80 + "\n")
            f.write(f"  待预测样本数(Reweight):   {results.get('n_vote', 0)}\n")
            f.write(f"  投票阈值:       {args.vote_k}\n")
            f.write("\n")
            f.write("  投票分布:\n")
            for i, count in enumerate(results.get('vote_distribution', [])):
                f.write(f"    {i}票判恶意:   {count} 个样本\n")
            f.write("\n")
            f.write("  各模型预测为恶意的比例:\n")
            for name, ratio in results.get('model_malicious_ratio', {}).items():
                f.write(f"    {name:<20} {ratio*100:>6.2f}%\n")
            f.write("\n")

            f.write("-"*80 + "\n")
            f.write("6. 评估 (预测 vs 真实标签)\n")
            f.write("-"*80 + "\n")
            f.write("  Baseline对比:\n")
            f.write(f"    y_noisy vs y_true:        {results.get('acc_noisy', 0.0):.4f}\n")
            f.write(f"    y_corrected_hc vs y_true: {results.get('acc_hc', 0.0):.4f}\n")
            f.write(f"    y_final vs y_true:        {results.get('correction_accuracy', 0.0):.4f}\n")
            f.write("\n")
            m_all = results.get('metrics_all', {})
            f.write("  全量评估:\n")
            f.write(f"    Accuracy:     {m_all.get('accuracy', 0.0):.4f}\n")
            f.write(f"    F1(pos=1):    {m_all.get('f1_pos', 0.0):.4f}\n")
            f.write(f"    Correction accuracy: {results.get('correction_accuracy', m_all.get('accuracy', 0.0)):.4f}\n")
            f.write(f"    Confusion:\n{m_all.get('confusion_matrix', '')}\n")
            f.write("\n")
            m_vote = results.get('metrics_vote', None)
            if m_vote is not None:
                f.write("  仅对投票矫正样本(Reweight)评估:\n")
                f.write(f"    Accuracy:     {m_vote.get('accuracy', 0.0):.4f}\n")
                f.write(f"    F1(pos=1):    {m_vote.get('f1_pos', 0.0):.4f}\n")
                f.write(f"    Confusion:\n{m_vote.get('confusion_matrix', '')}\n")
                if results.get('report_vote', ''):
                    f.write("\n")
                    f.write("  Classification Report (vote/reweight):\n")
                    f.write(results.get('report_vote', '') + "\n")
            f.write("\n")

            f.write("-"*80 + "\n")
            f.write("7. 输出文件\n")
            f.write("-"*80 + "\n")
            f.write(f"  分析报告:       {save_path}\n")
            f.write(f"  实验摘要:       {args.output_dir}/summary.json\n")
            if results.get('sample_report', ''):
                f.write(f"  样本分析CSV:    {results.get('sample_report', '')}\n")
            f.write(f"  质量分布图:     {args.output_dir}/figures/1_质量分数分布.png\n")
            f.write(f"  投票分析图:     {args.output_dir}/figures/2_投票分析.png\n")
            f.write(f"  特征空间图:     {args.output_dir}/figures/3_特征空间可视化.png\n")
            f.write(f"  标签文件:       {args.output_dir}/y_true.npy, y_noisy.npy, y_corrected_hc.npy, y_pred_all.npy\n")
            f.write("\n")
            f.write("="*80 + "\n")
            return
        
        # 实验配置
        f.write("-"*80 + "\n")
        f.write("1. 实验配置\n")
        f.write("-"*80 + "\n")
        f.write(f"  特征文件:       {args.features}\n")
        f.write(f"  矫正结果文件:   {args.correction}\n")
        f.write(f"  高质量样本比例: {args.hq_ratio} (legacy模式下已不作为划分依据)\n")
        f.write(f"  训练集口径:     {getattr(args, 'train_on', 'keep')}\n")
        f.write(f"  投票阈值:       {args.vote_k} (至少{args.vote_k}个模型同意才判为恶意)\n")
        f.write(f"  基础标签:       {args.use_base}\n")
        f.write(f"  类别平衡训练:   {'是' if args.balance_train else '否'}\n")
        f.write(f"  随机种子:       {args.seed}\n")
        f.write("\n")
        
        # 数据统计
        f.write("-"*80 + "\n")
        f.write("2. 数据统计\n")
        f.write("-"*80 + "\n")
        f.write(f"  总样本数:       {results['n_total']}\n")
        f.write(f"  高质量样本:     {results['n_high']} ({100*results['n_high']/results['n_total']:.1f}%)\n")
        f.write(f"  低质量样本:     {results['n_low']} ({100*results['n_low']/results['n_total']:.1f}%)\n")
        f.write(f"  Drop样本:       {results.get('n_drop', 0)} ({100*results.get('n_drop', 0)/results['n_total']:.1f}%)\n")
        f.write("\n")
        f.write(f"  原始标签分布:\n")
        f.write(f"    正常流量:     {results['original_benign']}\n")
        f.write(f"    恶意流量:     {results['original_malicious']}\n")
        f.write("\n")

        f.write("-"*80 + "\n")
        f.write("3. 标签矫正动作统计 (HybridCourt)\n")
        f.write("-"*80 + "\n")
        f.write(f"  Keep:           {results.get('action_keep', 0)}\n")
        f.write(f"  Flip:           {results.get('action_flip', 0)}\n")
        f.write(f"  Drop:           {results.get('action_drop', 0)}\n")
        f.write(f"  Reweight:       {results.get('action_reweight', 0)}\n")
        f.write("\n")
        
        # 质量分数统计
        f.write("-"*80 + "\n")
        f.write("4. 质量分数统计\n")
        f.write("-"*80 + "\n")
        q = results['quality']
        f.write(f"  最小值:         {q.min():.4f}\n")
        f.write(f"  最大值:         {q.max():.4f}\n")
        f.write(f"  均值:           {q.mean():.4f}\n")
        f.write(f"  中位数:         {np.median(q):.4f}\n")
        f.write(f"  标准差:         {q.std():.4f}\n")
        f.write("\n")
        f.write(f"  高质量样本质量分数: {results.get('high_quality_mean', 0.0):.4f} ± {results.get('high_quality_std', 0.0):.4f}\n")
        f.write(f"  低质量样本质量分数: {results.get('low_quality_mean', 0.0):.4f} ± {results.get('low_quality_std', 0.0):.4f}\n")
        f.write("\n")
        
        # 模型训练结果
        f.write("-"*80 + "\n")
        f.write("5. 模型训练结果\n")
        f.write("-"*80 + "\n")
        f.write(f"  训练样本数:     {results.get('train_samples', 0)}\n")
        f.write(f"  使用的模型:     {len(results.get('models', []))} 个\n")
        f.write("\n")
        f.write(f"  {'模型名称':<20} {'训练集准确率':>15}\n")
        f.write(f"  {'-'*20} {'-'*15}\n")
        for name, acc in results['model_accuracies'].items():
            f.write(f"  {name:<20} {acc*100:>14.2f}%\n")
        f.write("\n")
        
        # 投票结果
        f.write("-"*80 + "\n")
        f.write("6. 投票矫正结果\n")
        f.write("-"*80 + "\n")
        f.write(f"  低质量样本数:   {results['n_low']}\n")
        f.write(f"  投票阈值:       {args.vote_k}\n")
        f.write("\n")
        f.write(f"  投票分布:\n")
        for i, count in enumerate(results['vote_distribution']):
            f.write(f"    {i}票判恶意:   {count} 个样本\n")
        f.write("\n")
        f.write(f"  各模型预测为恶意的比例:\n")
        for name, ratio in results['model_malicious_ratio'].items():
            f.write(f"    {name:<20} {ratio*100:>6.2f}%\n")
        f.write("\n")
        
        # 矫正统计
        f.write("-"*80 + "\n")
        f.write("7. 矫正统计\n")
        f.write("-"*80 + "\n")
        f.write(f"  标签变化数:     {results['changed']} ({100*results['changed']/results['n_total']:.2f}%)\n")
        f.write("\n")
        f.write(f"  低质量样本矫正详情:\n")
        f.write(f"    保持正常:     {results['keep_benign']}\n")
        f.write(f"    正常→恶意:    {results['benign_to_malicious']}\n")
        f.write(f"    恶意→正常:    {results['malicious_to_benign']}\n")
        f.write(f"    保持恶意:     {results['keep_malicious']}\n")
        f.write("\n")
        f.write(f"  矫正后标签分布:\n")
        f.write(f"    正常流量:     {results['final_benign']}\n")
        f.write(f"    恶意流量:     {results['final_malicious']}\n")
        f.write("\n")

        # 矫正后分析（优先使用y_true；若无y_true则回退到与HC输出的一致率）
        f.write("-"*80 + "\n")
        f.write("8. 矫正后分析(一致率/混淆矩阵)\n")
        f.write("-"*80 + "\n")
        if results.get('has_y_true', False):
            f.write("  ✅ 已加载真实标签(y_true)，以下为真实评估指标(预测 vs 真实)：\n")
            f.write(f"  Correction accuracy: {results.get('correction_accuracy', results.get('acc_true', 0.0)):.4f}\n")
            f.write(f"  Accuracy:        {results.get('acc_true', 0.0):.4f}\n")
            f.write(f"  F1(pos=1):       {results.get('f1_true', 0.0):.4f}\n")
            if results.get('cm_true', ''):
                f.write(f"  Confusion Matrix (y_true vs y_final):\n{results.get('cm_true', '')}\n")
            f.write("\n")

            # 详细子集评估
            f.write("  子集评估(Accuracy/F1)：\n")
            for key, title in [
                ('metrics_all', 'All'),
                ('metrics_train', 'TrainSubset'),
                ('metrics_vote', 'VoteSubset'),
                ('metrics_changed', 'ChangedOnly'),
                ('metrics_keep', 'Action=Keep'),
                ('metrics_flip', 'Action=Flip'),
                ('metrics_reweight', 'Action=Reweight'),
                ('metrics_drop', 'Action=Drop'),
            ]:
                m = results.get(key, None)
                if isinstance(m, dict):
                    f.write(f"    {title:<14} acc={m.get('accuracy', 0.0):.4f}, f1(pos=1)={m.get('f1_pos', 0.0):.4f}\n")
            f.write("\n")

        f.write("  参考一致率(用于对齐HybridCourt输出/输入标签)：\n")
        f.write(f"    与 y_corrected(HC) 一致率: {results.get('acc_vs_hc', 0.0):.4f}\n")
        f.write(f"    与 y_noisy 一致率:         {results.get('acc_vs_noisy', 0.0):.4f}\n")
        f.write(f"    与 y_base 一致率:          {results.get('acc_vs_base', 0.0):.4f}\n")
        if results.get('cm_vs_hc', ''):
            f.write(f"    Confusion Matrix (y_corrected vs y_final):\n{results.get('cm_vs_hc', '')}\n")
        f.write("\n")
        
        # 输出文件
        f.write("-"*80 + "\n")
        f.write("9. 输出文件\n")
        f.write("-"*80 + "\n")
        f.write(f"  分析报告:       {save_path}\n")
        f.write(f"  实验摘要:       {args.output_dir}/summary.json\n")
        f.write(f"  质量分布图:     {args.output_dir}/figures/1_质量分数分布.png\n")
        f.write(f"  投票分析图:     {args.output_dir}/figures/2_投票分析.png\n")
        f.write(f"  特征空间图:     {args.output_dir}/figures/3_特征空间可视化.png\n")
        f.write(f"  矫正后标签:     {args.output_dir}/y_corrected_ml.npy\n")
        if results.get('sample_report', ''):
            f.write(f"  样本分析CSV:    {results.get('sample_report', '')}\n")
        f.write("\n")
        f.write("="*80 + "\n")
    
    logger.info(f"  ✓ 已保存: {save_path}")


def main():
    ap = argparse.ArgumentParser(description="ML投票标签矫正实验")
    ap.add_argument("--mode", type=str, default="legacy", choices=["legacy", "from_preprocessed"], help="运行模式")
    ap.add_argument("--data_split", type=str, default="train", choices=["train", "test"], help="预处理数据切分")
    ap.add_argument("--noise_rate", type=float, default=0.0, help="注入标签噪声率(用于可控评估)，0表示不注入")
    ap.add_argument("--features_npy", type=str, default="", help="(可选) 直接提供(N,d)特征npy，跳过backbone特征提取")
    ap.add_argument("--backbone", type=str, default="", help="(可选) backbone权重路径，默认使用output/feature_extraction/models/backbone_pretrained.pth")
    ap.add_argument("--hc_impl", type=str, default="experimental", choices=["experimental", "original"], help="HybridCourt实现选择: experimental=复制版(可修改), original=原版(不改动)")
    ap.add_argument("--skip_plots", action="store_true", help="跳过绘图与t-SNE（用于自动调参加速）")
    ap.add_argument("--save_cache_dir", type=str, default="", help="(可选) 保存from_preprocessed阶段的features/HC输出到缓存目录（用于自动调参复用）")
    ap.add_argument("--reuse_cache_dir", type=str, default="", help="(可选) 复用缓存目录，跳过特征提取与HybridCourt")
    ap.add_argument("--train_on", type=str, default="keep", choices=["keep", "keep_flip", "nondrop"], help="训练集口径: keep=action==0, keep_flip=action in {0,1}, nondrop=action!=2")
    ap.add_argument("--features", type=str, default="./output/feature_extraction/models/train_features.npy")
    ap.add_argument("--correction", type=str, default="./output/label_correction/models/correction_results.npz")
    ap.add_argument("--output_dir", type=str, default="./experiments/ml_vote_label_correction/output")
    ap.add_argument("--hq_ratio", type=float, default=0.5, help="高质量样本比例")
    ap.add_argument("--balance_train", action="store_true", help="启用类别平衡训练")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_base", type=str, default="y_corrected", choices=["y_corrected", "y_noisy"])
    ap.add_argument("--vote_k", type=int, default=4, help="投票阈值")
    ap.add_argument("--vote_rule", type=str, default="hard", choices=["hard", "band", "none"], help="投票覆盖策略: hard=全覆盖, band=高置信覆盖, none=不覆盖")
    ap.add_argument("--band_low", type=int, default=-1, help="band策略: vote_sum<=band_low 判为正常并覆盖")
    ap.add_argument("--band_high", type=int, default=8, help="band策略: vote_sum>=band_high 判为恶意并覆盖")
    ap.add_argument("--vote_include_drop", action="store_true", help="允许对Drop(action=2)样本也进行投票/概率高置信覆盖")
    ap.add_argument("--flip_train_weight", type=float, default=0.5, help="train_on=keep_flip时，对Flip(action=1)样本的训练权重额外乘以该系数")
    ap.add_argument("--vote_score", type=str, default="count", choices=["count", "mean_proba", "stacked", "mlp"], help="投票打分方式: count=票数, mean_proba=平均概率, stacked=stacking概率, mlp=MLP概率")
    ap.add_argument("--prob_threshold", type=float, default=0.5, help="概率硬覆盖阈值(score>=th判恶意)，仅对vote_score!=count生效")
    ap.add_argument("--prob_low", type=float, default=0.3, help="band策略(概率): score<=prob_low 判正常并覆盖")
    ap.add_argument("--prob_high", type=float, default=0.7, help="band策略(概率): score>=prob_high 判恶意并覆盖")
    ap.add_argument("--mlp_hidden", type=int, default=128, help="MLP hidden size (vote_score=mlp)")
    ap.add_argument("--mlp_epochs", type=int, default=200, help="MLP epochs (vote_score=mlp)")
    ap.add_argument("--mlp_lr", type=float, default=1e-3, help="MLP learning rate (vote_score=mlp)")
    ap.add_argument("--mlp_weight_decay", type=float, default=1e-4, help="MLP weight decay (vote_score=mlp)")
    args = ap.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    logger.info("="*70)
    logger.info("ML投票标签矫正实验")
    logger.info("="*70)
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    if args.mode == "from_preprocessed":
        logger.info("步骤1: 加载预处理数据")
        logger.info("-"*70)
        _run_from_preprocessed(args, logger)
        return

    # 加载数据
    logger.info("步骤1: 加载数据")
    logger.info("-"*70)
    
    if not os.path.exists(args.features):
        logger.error(f"❌ 特征文件不存在: {args.features}")
        return
    if not os.path.exists(args.correction):
        logger.error(f"❌ 矫正结果文件不存在: {args.correction}")
        return

    features = np.load(args.features)
    corr = np.load(args.correction, allow_pickle=True)
    
    logger.info(f"  特征形状: {features.shape}")

    required = ["y_noisy", "y_corrected", "correction_weight", "density_scores", "neighbor_consistency", "pred_probs"]
    missing = [k for k in required if k not in corr]
    if missing:
        logger.error(f"❌ 缺少必要字段: {missing}")
        return

    y_noisy = corr["y_noisy"].astype(int)
    y_corrected = corr["y_corrected"].astype(int)
    y_base = y_corrected if args.use_base == "y_corrected" else y_noisy

    # 尝试加载真实标签(y_true)：参考label_correction_analysis，通过预处理数据获取
    y_true = None
    if PREPROCESS_AVAILABLE and check_preprocessed_exists('train'):
        try:
            _, y_tmp, _ = load_preprocessed('train')
            y_tmp = y_tmp.astype(int)
            if len(y_tmp) == len(y_base):
                y_true = y_tmp
            else:
                # 如果train不匹配，再尝试test
                if check_preprocessed_exists('test'):
                    _, y_tmp2, _ = load_preprocessed('test')
                    y_tmp2 = y_tmp2.astype(int)
                    if len(y_tmp2) == len(y_base):
                        y_true = y_tmp2
        except Exception:
            y_true = None

    w = corr["correction_weight"].astype(np.float32)
    density = corr["density_scores"].astype(np.float32)
    knn_cons = corr["neighbor_consistency"].astype(np.float32)
    cl_conf = _softmax_max_prob(corr["pred_probs"]).astype(np.float32)

    logger.info(f"  总样本数: {len(y_base)}")
    logger.info(f"  正常流量: {(y_base == 0).sum()}")
    logger.info(f"  恶意流量: {(y_base == 1).sum()}")
    logger.info("")

    # 计算质量分数
    logger.info("步骤2: 计算质量分数")
    logger.info("-"*70)
    
    density_norm = _minmax01(density)
    knn_cons = np.clip(knn_cons, 0.0, 1.0)
    cl_conf = np.clip(cl_conf, 0.0, 1.0)
    w = np.clip(w, 0.0, None)

    quality = (0.5 * density_norm + 0.5 * knn_cons) * cl_conf
    quality = quality * np.maximum(w, 1e-6)
    
    logger.info(f"  质量分数范围: [{quality.min():.4f}, {quality.max():.4f}]")
    logger.info(f"  质量分数均值: {quality.mean():.4f}")
    logger.info("")

    # 使用标签矫正模块产物(action_mask)初筛高/低质量数据集
    logger.info("步骤3: 基于标签矫正动作划分高/低质量数据集")
    logger.info("-"*70)

    if "action_mask" not in corr:
        logger.error("❌ correction_results.npz 缺少 action_mask，无法按标签矫正初筛数据。")
        return

    rng = np.random.default_rng(args.seed)
    action_mask = corr["action_mask"].astype(int)

    # action 约定: 0=Keep, 1=Flip, 2=Drop, 3=Reweight
    idx_keep = np.where(action_mask == 0)[0]
    idx_flip = np.where(action_mask == 1)[0]
    idx_drop = np.where(action_mask == 2)[0]
    idx_reweight = np.where(action_mask == 3)[0]

    if args.train_on == "keep":
        train_mask = action_mask == 0
    else:
        train_mask = action_mask != 2

    idx_high = np.where(train_mask)[0]
    # 低质量集合：不进入训练集且非drop(默认不对drop投票)
    idx_low = np.where((~train_mask) & (action_mask != 2))[0]

    logger.info(f"  train_on: {args.train_on}")
    logger.info(f"  Keep:     {len(idx_keep)}")
    logger.info(f"  Flip:     {len(idx_flip)}")
    logger.info(f"  Drop:     {len(idx_drop)}")
    logger.info(f"  Reweight: {len(idx_reweight)}")
    logger.info(f"  高质量(训练)样本: {len(idx_high)} ({100*len(idx_high)/len(y_base):.1f}%)")
    logger.info(f"  低质量(投票)样本: {len(idx_low)} ({100*len(idx_low)/len(y_base):.1f}%)")
    logger.info("")

    if len(idx_high) == 0:
        logger.error("❌ 高质量训练集为空（按action_mask筛选）。请检查标签矫正结果或尝试 --train_on nondrop")
        return

    # 准备训练数据
    X_train = features[idx_high]
    y_train = y_base[idx_high]

    if args.balance_train:
        logger.info("  启用类别平衡训练...")
        idx0 = idx_high[y_base[idx_high] == 0]
        idx1 = idx_high[y_base[idx_high] == 1]
        n = min(len(idx0), len(idx1))
        if n > 0:
            idx0_s = rng.choice(idx0, size=n, replace=False)
            idx1_s = rng.choice(idx1, size=n, replace=False)
            idx_bal = np.concatenate([idx0_s, idx1_s])
            rng.shuffle(idx_bal)
            X_train = features[idx_bal]
            y_train = y_base[idx_bal]
            logger.info(f"  平衡后训练样本: {len(X_train)} (每类 {n})")

    # 训练模型
    logger.info("")
    logger.info("步骤4: 训练ML模型")
    logger.info("-"*70)
    
    models = _build_models(args.seed)
    logger.info(f"  使用 {len(models)} 个模型: {list(models.keys())}")
    logger.info("")
    
    fitted, accuracies = _fit_models(models, X_train, y_train, logger)
    logger.info("")

    # 对低质量样本进行投票
    logger.info("步骤5: 对低质量样本进行投票矫正")
    logger.info("-"*70)
    
    X_low = features[idx_low]
    vote_mat, model_names = _vote_predict(fitted, X_low)
    vote_sum = vote_mat.sum(axis=1) if vote_mat.size else np.zeros((len(X_low),), dtype=int)
    y_low_pred = (vote_sum >= args.vote_k).astype(int)

    # 统计投票分布
    vote_distribution = [0] * (len(model_names) + 1)
    for v in vote_sum:
        vote_distribution[v] += 1
    
    logger.info(f"  投票阈值: {args.vote_k}")
    logger.info(f"  投票分布:")
    for i, count in enumerate(vote_distribution):
        logger.info(f"    {i}票判恶意: {count} 个样本")
    logger.info("")

    # 生成最终标签
    y_final = y_base.copy()
    y_final[idx_low] = y_low_pred

    # legacy模式的“矫正后分析”：由于缺少y_true，计算与HC/噪声/基础标签的一致率
    acc_vs_hc = float(np.mean(y_final == y_corrected))
    acc_vs_noisy = float(np.mean(y_final == y_noisy))
    acc_vs_base = float(np.mean(y_final == y_base))
    cm_vs_hc = confusion_matrix(y_corrected, y_final, labels=[0, 1])
    cm_vs_hc_str = np.array2string(cm_vs_hc)

    has_y_true = y_true is not None
    acc_true = None
    f1_true = None
    cm_true_str = ""
    metrics_all = None
    metrics_train = None
    metrics_vote = None
    metrics_changed = None
    metrics_keep = None
    metrics_flip = None
    metrics_reweight = None
    metrics_drop = None
    if has_y_true:
        # 全量指标
        metrics_all = calculate_metrics(y_true, y_final)
        acc_true = float(metrics_all.get('accuracy', 0.0))
        f1_true = float(metrics_all.get('f1_pos', 0.0))
        cm_true = confusion_matrix(y_true, y_final, labels=[0, 1])
        cm_true_str = np.array2string(cm_true)

        # 子集指标
        if len(idx_high) > 0:
            metrics_train = calculate_metrics(y_true[idx_high], y_final[idx_high])
        if len(idx_low) > 0:
            metrics_vote = calculate_metrics(y_true[idx_low], y_final[idx_low])
        changed_idx = np.where(y_final != y_base)[0]
        if len(changed_idx) > 0:
            metrics_changed = calculate_metrics(y_true[changed_idx], y_final[changed_idx])
        if len(idx_keep) > 0:
            metrics_keep = calculate_metrics(y_true[idx_keep], y_final[idx_keep])
        if len(idx_flip) > 0:
            metrics_flip = calculate_metrics(y_true[idx_flip], y_final[idx_flip])
        if len(idx_reweight) > 0:
            metrics_reweight = calculate_metrics(y_true[idx_reweight], y_final[idx_reweight])
        if len(idx_drop) > 0:
            metrics_drop = calculate_metrics(y_true[idx_drop], y_final[idx_drop])

    # 统计变化
    changed = int(np.sum(y_final != y_base))
    original_low = y_base[idx_low]
    keep_benign = ((original_low == 0) & (y_low_pred == 0)).sum()
    benign_to_malicious = ((original_low == 0) & (y_low_pred == 1)).sum()
    malicious_to_benign = ((original_low == 1) & (y_low_pred == 0)).sum()
    keep_malicious = ((original_low == 1) & (y_low_pred == 1)).sum()

    logger.info(f"  标签变化统计:")
    logger.info(f"    保持正常:     {keep_benign}")
    logger.info(f"    正常→恶意:    {benign_to_malicious}")
    logger.info(f"    恶意→正常:    {malicious_to_benign}")
    logger.info(f"    保持恶意:     {keep_malicious}")
    logger.info(f"    总变化数:     {changed} ({100*changed/len(y_base):.2f}%)")
    logger.info("")

    # 收集结果
    model_malicious_ratio = {}
    for i, name in enumerate(model_names):
        model_malicious_ratio[name] = vote_mat[:, i].mean()

    results = {
        'n_total': len(y_base),
        'n_high': len(idx_high),
        'n_low': len(idx_low),
        'n_drop': int(len(idx_drop)),
        'train_on': str(args.train_on),
        'action_keep': int(len(idx_keep)),
        'action_flip': int(len(idx_flip)),
        'action_drop': int(len(idx_drop)),
        'action_reweight': int(len(idx_reweight)),
        'original_benign': int((y_base == 0).sum()),
        'original_malicious': int((y_base == 1).sum()),
        'final_benign': int((y_final == 0).sum()),
        'final_malicious': int((y_final == 1).sum()),
        'quality': quality,
        'high_quality_mean': float(quality[idx_high].mean()),
        'high_quality_std': float(quality[idx_high].std()),
        'low_quality_mean': float(quality[idx_low].mean()) if len(idx_low) > 0 else 0,
        'low_quality_std': float(quality[idx_low].std()) if len(idx_low) > 0 else 0,
        'train_samples': len(X_train),
        'models': list(fitted.keys()),
        'model_accuracies': accuracies,
        'vote_distribution': vote_distribution,
        'model_malicious_ratio': model_malicious_ratio,
        'changed': changed,
        'keep_benign': int(keep_benign),
        'benign_to_malicious': int(benign_to_malicious),
        'malicious_to_benign': int(malicious_to_benign),
        'keep_malicious': int(keep_malicious),
        'acc_vs_hc': acc_vs_hc,
        'acc_vs_noisy': acc_vs_noisy,
        'acc_vs_base': acc_vs_base,
        'cm_vs_hc': cm_vs_hc_str,
        'has_y_true': bool(has_y_true),
        'acc_true': float(acc_true) if acc_true is not None else 0.0,
        'f1_true': float(f1_true) if f1_true is not None else 0.0,
        'correction_accuracy': float(acc_true) if acc_true is not None else 0.0,
        'cm_true': cm_true_str,
        'metrics_all': metrics_all,
        'metrics_train': metrics_train,
        'metrics_vote': metrics_vote,
        'metrics_changed': metrics_changed,
        'metrics_keep': metrics_keep,
        'metrics_flip': metrics_flip,
        'metrics_reweight': metrics_reweight,
        'metrics_drop': metrics_drop,
    }

    # 逐样本分析报告（CSV）
    sample_report_path = os.path.join(args.output_dir, "sample_analysis.csv")
    vote_sum_all = np.full((len(y_base),), -1, dtype=np.int32)
    vote_pred_all = np.full((len(y_base),), -1, dtype=np.int32)
    vote_sum_all[idx_low] = vote_sum.astype(np.int32)
    vote_pred_all[idx_low] = y_low_pred.astype(np.int32)

    model_vote_all = {}
    for mi, name in enumerate(model_names):
        arr = np.full((len(y_base),), -1, dtype=np.int32)
        if vote_mat.size:
            arr[idx_low] = vote_mat[:, mi].astype(np.int32)
        model_vote_all[name] = arr

    in_train = np.zeros((len(y_base),), dtype=np.int32)
    in_vote = np.zeros((len(y_base),), dtype=np.int32)
    in_train[idx_high] = 1
    in_vote[idx_low] = 1

    with open(sample_report_path, "w", encoding="utf-8", newline="") as fcsv:
        fieldnames = [
            "index",
            "y_true",
            "y_noisy",
            "y_corrected_hc",
            "y_base",
            "y_final",
            "action_mask",
            "correction_weight",
            "density_score",
            "knn_consistency",
            "cl_conf",
            "quality",
            "in_train",
            "in_vote",
            "vote_sum",
            "vote_pred",
        ] + [f"vote_{n}" for n in model_names]
        wcsv = csv.DictWriter(fcsv, fieldnames=fieldnames)
        wcsv.writeheader()
        for i in range(len(y_base)):
            row = {
                "index": int(i),
                "y_true": int(y_true[i]) if has_y_true else -1,
                "y_noisy": int(y_noisy[i]),
                "y_corrected_hc": int(y_corrected[i]),
                "y_base": int(y_base[i]),
                "y_final": int(y_final[i]),
                "action_mask": int(action_mask[i]),
                "correction_weight": float(w[i]),
                "density_score": float(density[i]),
                "knn_consistency": float(knn_cons[i]),
                "cl_conf": float(cl_conf[i]),
                "quality": float(quality[i]),
                "in_train": int(in_train[i]),
                "in_vote": int(in_vote[i]),
                "vote_sum": int(vote_sum_all[i]),
                "vote_pred": int(vote_pred_all[i]),
            }
            for n in model_names:
                row[f"vote_{n}"] = int(model_vote_all[n][i])
            wcsv.writerow(row)

    results["sample_report"] = sample_report_path

    # 生成可视化
    logger.info("步骤6: 生成可视化图表")
    logger.info("-"*70)
    
    fig_dir = os.path.join(args.output_dir, "figures")
    
    plot_quality_distribution(
        quality, y_base, idx_high, idx_low,
        os.path.join(fig_dir, "1_质量分数分布.png"), logger
    )
    
    plot_vote_analysis(
        vote_mat, vote_sum, y_low_pred, idx_low, y_base, model_names,
        os.path.join(fig_dir, "2_投票分析.png"), logger
    )
    
    plot_feature_space(
        features, y_base, y_final, idx_high, idx_low,
        os.path.join(fig_dir, "3_特征空间可视化.png"), logger
    )
    logger.info("")

    # 生成报告
    logger.info("步骤7: 生成分析报告")
    logger.info("-"*70)
    
    generate_report(args, results, os.path.join(args.output_dir, "report.txt"), logger)
    logger.info("")

    # 保存数据文件
    logger.info("步骤8: 保存数据文件")
    logger.info("-"*70)

    # 保存摘要JSON
    summary = {
        "n_total": int(len(y_base)),
        "n_high": int(len(idx_high)),
        "n_low": int(len(idx_low)),
        "n_drop": int(len(idx_drop)),
        "hq_ratio": float(args.hq_ratio),
        "train_on": str(args.train_on),
        "balance_train": bool(args.balance_train),
        "vote_k": int(args.vote_k),
        "changed": changed,
        "base_label": args.use_base,
        "models": list(fitted.keys()),
        "original_benign": int((y_base == 0).sum()),
        "original_malicious": int((y_base == 1).sum()),
        "final_benign": int((y_final == 0).sum()),
        "final_malicious": int((y_final == 1).sum()),
        "acc_vs_hc": float(acc_vs_hc),
        "acc_vs_noisy": float(acc_vs_noisy),
        "acc_vs_base": float(acc_vs_base),
        "has_y_true": bool(has_y_true),
        "acc_true": float(acc_true) if acc_true is not None else None,
        "f1_true": float(f1_true) if f1_true is not None else None,
        "correction_accuracy": float(acc_true) if acc_true is not None else None,
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"  ✓ summary.json")

    np.save(os.path.join(args.output_dir, "y_base.npy"), y_base)
    np.save(os.path.join(args.output_dir, "y_corrected_ml.npy"), y_final)
    np.save(os.path.join(args.output_dir, "quality_score.npy"), quality)
    logger.info(f"  ✓ y_base.npy, y_corrected_ml.npy, quality_score.npy")
    logger.info("")

    # 最终总结
    logger.info("="*70)
    logger.info("🎉 实验完成!")
    logger.info("="*70)
    logger.info("")
    logger.info("📊 结果摘要:")
    logger.info(f"  总样本数:       {len(y_base)}")
    logger.info(f"  高质量样本:     {len(idx_high)} ({100*len(idx_high)/len(y_base):.1f}%)")
    logger.info(f"  低质量样本:     {len(idx_low)} ({100*len(idx_low)/len(y_base):.1f}%)")
    logger.info(f"  标签变化数:     {changed} ({100*changed/len(y_base):.2f}%)")
    if has_y_true:
        logger.info(f"  Correction accuracy: {acc_true:.4f}")
    logger.info("")
    logger.info(f"  原始分布:       正常={results['original_benign']}, 恶意={results['original_malicious']}")
    logger.info(f"  矫正后分布:     正常={results['final_benign']}, 恶意={results['final_malicious']}")
    logger.info("")
    logger.info("📁 输出文件:")
    logger.info(f"  {args.output_dir}/report.txt")
    logger.info(f"  {args.output_dir}/figures/1_质量分数分布.png")
    logger.info(f"  {args.output_dir}/figures/2_投票分析.png")
    logger.info(f"  {args.output_dir}/figures/3_特征空间可视化.png")
    logger.info("="*70)


if __name__ == "__main__":
    main()
