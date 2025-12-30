#!/usr/bin/env python3
"""
GPU检测脚本
检查PyTorch是否能正确使用GPU
"""

import sys
import os
from pathlib import Path

# Ensure project root is on sys.path when running as a script
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

def check_gpu():
    """检测GPU可用性"""
    print("="*60)
    print("GPU 可用性检测")
    print("="*60)
    
    try:
        import torch
        
        print("\n1. PyTorch 版本信息:")
        print(f"   PyTorch 版本: {torch.__version__}")
        
        # 检查CUDA
        print("\n2. CUDA 可用性:")
        if torch.cuda.is_available():
            print("   ✅ CUDA 可用")
            print(f"   CUDA 版本: {torch.version.cuda}")
            print(f"   cuDNN 版本: {torch.backends.cudnn.version()}")
            
            # GPU信息
            print("\n3. GPU 设备信息:")
            gpu_count = torch.cuda.device_count()
            print(f"   可用GPU数量: {gpu_count}")
            
            for i in range(gpu_count):
                print(f"\n   GPU {i}:")
                print(f"     名称: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"     计算能力: {props.major}.{props.minor}")
                print(f"     显存总量: {props.total_memory / 1024**3:.2f} GB")
                
                # 当前显存使用
                if torch.cuda.is_initialized():
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    cached = torch.cuda.memory_reserved(i) / 1024**3
                    print(f"     已分配: {allocated:.2f} GB")
                    print(f"     已缓存: {cached:.2f} GB")
            
            # 测试GPU计算
            print("\n4. GPU 计算测试:")
            try:
                device = torch.device("cuda:0")
                x = torch.randn(1000, 1000, device=device)
                y = torch.randn(1000, 1000, device=device)
                z = torch.matmul(x, y)
                print("   ✅ GPU 计算测试通过")
                print(f"   测试矩阵: {x.shape} @ {y.shape} = {z.shape}")
                print(f"   结果设备: {z.device}")
            except Exception as e:
                print(f"   ❌ GPU 计算测试失败: {e}")
                return False
            
            # 测试配置
            print("\n5. 项目配置:")
            try:
                from MoudleCode.utils.config import config
                print(f"   配置设备: {config.DEVICE}")
                if str(config.DEVICE) == "cuda":
                    print("   ✅ 项目配置正确，将使用GPU加速")
                else:
                    print(f"   ⚠️  项目配置为 {config.DEVICE}")
            except Exception as e:
                print(f"   ⚠️  无法加载项目配置: {e}")
            
            print("\n" + "="*60)
            print("✅ GPU完全可用，可以进行训练")
            print("="*60)
            
            # 性能建议
            print("\n💡 性能优化建议:")
            print("   1. 使用 CUDA 11.x 或更高版本以获得最佳性能")
            print("   2. 启用 cuDNN 自动调优:")
            print("      torch.backends.cudnn.benchmark = True")
            print("   3. 监控显存使用，避免OOM错误")
            print("   4. 建议批次大小: 64 (可根据显存调整)")
            
            return True
            
        else:
            print("   ❌ CUDA 不可用")
            print("\n可能的原因:")
            print("   1. 未安装CUDA")
            print("   2. 未安装GPU版本的PyTorch")
            print("   3. GPU驱动问题")
            print("\n解决方案:")
            print("   1. 安装CUDA: https://developer.nvidia.com/cuda-downloads")
            print("   2. 安装GPU版PyTorch:")
            print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            print("   3. 检查NVIDIA驱动:")
            print("      nvidia-smi")
            
            print("\n⚠️  当前将使用CPU训练（速度较慢）")
            return False
            
    except ImportError:
        print("\n❌ 错误: PyTorch未安装")
        print("请安装PyTorch:")
        print("   pip install torch torchvision")
        return False
    
    except Exception as e:
        print(f"\n❌ 检测失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_training_estimate():
    """显示训练时间估算"""
    print("\n" + "="*60)
    print("训练时间估算 (基于实际数据量，流数在处理时统计)")
    print("="*60)

    try:
        import torch
    except ImportError:
        print("\n⚠️  无法估算训练时间：PyTorch未安装")
        return
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        
        # 根据GPU类型估算
        if "RTX 2080" in gpu_name or "RTX 2070" in gpu_name:
            print("\n使用 GPU (RTX 20系列):")
            print("  Stage 1 (预训练):    2-3 小时")
            print("  Stage 2 (矫正+增强): 1-2 小时")
            print("  Stage 3 (微调):      30 分钟")
            print("  总计:               约 4-6 小时")
        elif "RTX 30" in gpu_name or "RTX 40" in gpu_name:
            print("\n使用 GPU (RTX 30/40系列):")
            print("  Stage 1 (预训练):    1-2 小时")
            print("  Stage 2 (矫正+增强): 0.5-1 小时")
            print("  Stage 3 (微调):      15-20 分钟")
            print("  总计:               约 2-4 小时")
        elif "Tesla" in gpu_name or "A100" in gpu_name:
            print("\n使用 GPU (数据中心级):")
            print("  Stage 1 (预训练):    0.5-1 小时")
            print("  Stage 2 (矫正+增强): 0.5 小时")
            print("  Stage 3 (微调):      10 分钟")
            print("  总计:               约 1-2 小时")
        else:
            print("\n使用 GPU:")
            print("  预计总时间: 3-6 小时")
    else:
        print("\n使用 CPU:")
        print("  Stage 1 (预训练):    8-10 小时")
        print("  Stage 2 (矫正+增强): 3-5 小时")
        print("  Stage 3 (微调):      1-2 小时")
        print("  总计:               约 12-17 小时")
        print("\n⚠️  强烈建议使用GPU以加速训练")


if __name__ == "__main__":
    has_gpu = check_gpu()
    show_training_estimate()
    
    print("\n" + "="*60)
    if has_gpu:
        print("✅ 系统就绪，可以开始训练")
        print("\n运行训练:")
        print("   cd /home/lx/python/MEDAL")
        print("   python all_train_test.py")
    else:
        print("⚠️  建议配置GPU后再进行训练")
        print("   或者调整参数以适应CPU训练:")
        print("   - 减少训练轮数")
        print("   - 减小批次大小")
        print("   - 减少训练样本数")
    print("="*60)
    
    sys.exit(0 if has_gpu else 1)

