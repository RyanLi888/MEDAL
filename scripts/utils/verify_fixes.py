#!/usr/bin/env python3
"""
验证修复脚本
检查 2026-01-09 的两个关键修复是否生效
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def test_config_import():
    """测试 config.py 的变量定义顺序修复"""
    print("=" * 60)
    print("测试 1: config.py 变量定义顺序")
    print("=" * 60)
    
    try:
        from MoudleCode.utils.config import config
        print("✓ config 模块导入成功")
        
        # 验证关键变量
        assert hasattr(config, 'SEQUENCE_LENGTH'), "SEQUENCE_LENGTH 未定义"
        assert config.SEQUENCE_LENGTH == 1024, f"SEQUENCE_LENGTH 值错误: {config.SEQUENCE_LENGTH}"
        print(f"✓ SEQUENCE_LENGTH = {config.SEQUENCE_LENGTH}")
        
        assert hasattr(config, 'BURSTSIZE_NORM_DENOM'), "BURSTSIZE_NORM_DENOM 未定义"
        assert config.BURSTSIZE_NORM_DENOM > 0, f"BURSTSIZE_NORM_DENOM 值异常: {config.BURSTSIZE_NORM_DENOM}"
        print(f"✓ BURSTSIZE_NORM_DENOM = {config.BURSTSIZE_NORM_DENOM:.4f}")
        
        # 验证计算正确性
        import math
        expected = math.log1p(config.MTU * config.SEQUENCE_LENGTH)
        assert abs(config.BURSTSIZE_NORM_DENOM - expected) < 1e-6, "BURSTSIZE_NORM_DENOM 计算错误"
        print(f"✓ BURSTSIZE_NORM_DENOM 计算正确 (log1p({config.MTU} * {config.SEQUENCE_LENGTH}))")
        
        print("\n✅ 测试 1 通过: config.py 修复成功\n")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试 1 失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_infonce_loss_import():
    """测试 InfoNCELoss 类定义修复"""
    print("=" * 60)
    print("测试 2: InfoNCELoss 类定义")
    print("=" * 60)
    
    try:
        from MoudleCode.feature_extraction.instance_contrastive import InfoNCELoss
        print("✓ InfoNCELoss 类导入成功")
        
        # 实例化测试
        loss_fn = InfoNCELoss(temperature=0.2)
        print(f"✓ InfoNCELoss 实例化成功 (temperature={loss_fn.temperature})")
        
        # 验证是否是 nn.Module
        import torch.nn as nn
        assert isinstance(loss_fn, nn.Module), "InfoNCELoss 不是 nn.Module 的子类"
        print("✓ InfoNCELoss 是 nn.Module 的子类")
        
        # 验证 forward 方法存在
        assert hasattr(loss_fn, 'forward'), "InfoNCELoss 缺少 forward 方法"
        print("✓ InfoNCELoss 具有 forward 方法")
        
        print("\n✅ 测试 2 通过: InfoNCELoss 修复成功\n")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试 2 失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_dual_stream_consistency_loss():
    """测试 DualStreamConsistencyLoss 类"""
    print("=" * 60)
    print("测试 3: DualStreamConsistencyLoss 类")
    print("=" * 60)
    
    try:
        from MoudleCode.feature_extraction.instance_contrastive import DualStreamConsistencyLoss
        print("✓ DualStreamConsistencyLoss 类导入成功")
        
        # 实例化测试
        loss_fn = DualStreamConsistencyLoss(temperature=0.2)
        print(f"✓ DualStreamConsistencyLoss 实例化成功 (temperature={loss_fn.temperature})")
        
        # 验证是否是 nn.Module
        import torch.nn as nn
        assert isinstance(loss_fn, nn.Module), "DualStreamConsistencyLoss 不是 nn.Module 的子类"
        print("✓ DualStreamConsistencyLoss 是 nn.Module 的子类")
        
        print("\n✅ 测试 3 通过: DualStreamConsistencyLoss 正常\n")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试 3 失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_instance_contrastive_learning():
    """测试 InstanceContrastiveLearning 类实例化"""
    print("=" * 60)
    print("测试 4: InstanceContrastiveLearning 实例化")
    print("=" * 60)
    
    try:
        from MoudleCode.feature_extraction.instance_contrastive import InstanceContrastiveLearning
        from MoudleCode.utils.config import config
        print("✓ 模块导入成功")
        
        # 创建一个简单的 mock backbone
        import torch.nn as nn
        class MockBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 32)
            
            def forward(self, x, return_sequence=False):
                # x: (B, L, D) -> (B, 32)
                return self.linear(x.mean(dim=1))
        
        backbone = MockBackbone()
        print("✓ Mock backbone 创建成功")
        
        # 实例化 InstanceContrastiveLearning
        icl = InstanceContrastiveLearning(backbone, config)
        print("✓ InstanceContrastiveLearning 实例化成功")
        
        # 验证 infonce_loss 属性存在
        assert hasattr(icl, 'infonce_loss'), "InstanceContrastiveLearning 缺少 infonce_loss 属性"
        print("✓ infonce_loss 属性存在")
        
        # 验证 infonce_loss 是 InfoNCELoss 实例
        from MoudleCode.feature_extraction.instance_contrastive import InfoNCELoss
        assert isinstance(icl.infonce_loss, InfoNCELoss), "infonce_loss 不是 InfoNCELoss 实例"
        print("✓ infonce_loss 是 InfoNCELoss 实例")
        
        print("\n✅ 测试 4 通过: InstanceContrastiveLearning 正常\n")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试 4 失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("MEDAL 修复验证脚本")
    print("修复日期: 2026-01-09")
    print("=" * 60 + "\n")
    
    results = []
    
    # 运行测试
    results.append(("config.py 变量定义顺序", test_config_import()))
    results.append(("InfoNCELoss 类定义", test_infonce_loss_import()))
    results.append(("DualStreamConsistencyLoss", test_dual_stream_consistency_loss()))
    results.append(("InstanceContrastiveLearning", test_instance_contrastive_learning()))
    
    # 汇总结果
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
    
    print("\n" + "=" * 60)
    print(f"总计: {passed}/{total} 测试通过")
    print("=" * 60 + "\n")
    
    if passed == total:
        print("🎉 所有修复验证通过！系统可以正常运行。\n")
        return 0
    else:
        print("⚠️  部分测试失败，请检查修复是否完整。\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
