"""
快速评估脚本 - 简化版
适合快速测试和调试
"""

from evaluate_backbone import BackboneEvaluator

# 配置参数（根据你的实际路径修改）
CONFIG = {
    'backbone_path': './output/stage1/best_model.pth',
    'data_root': './data/medical_images',
    'clean_labels_path': './data/clean_labels.json',
    'output_dir': './output/backbone_eval',
    'device': 'cuda'
}

def main():
    print("\n" + "="*70)
    print("骨干网络快速评估")
    print("="*70)
    
    # 显示配置
    print("\n配置信息:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    try:
        # 创建评估器
        evaluator = BackboneEvaluator(
            backbone_path=CONFIG['backbone_path'],
            data_root=CONFIG['data_root'],
            clean_labels_path=CONFIG['clean_labels_path'],
            device=CONFIG['device']
        )
        
        # 运行完整评估
        report = evaluator.run_full_evaluation(save_dir=CONFIG['output_dir'])
        
        # 打印最终建议
        print("\n" + "="*70)
        if report['decision']['need_supcon']:
            print("⚠️  下一步: 运行 Stage 2.5 (SupCon 微调)")
            print("   命令: python supcon_finetune.py --config configs/supcon.yaml")
        else:
            print("✓ 下一步: 直接进入 Stage 3 (双流 MiniMLP)")
            print("   命令: python stage3_dual_stream.py --config configs/stage3.yaml")
        print("="*70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误: 找不到文件 - {e}")
        print("\n请检查 CONFIG 中的路径设置:")
        print("  1. backbone_path: Stage1 训练好的模型权重")
        print("  2. data_root: 数据集根目录")
        print("  3. clean_labels_path: 真值标签 JSON 文件")
        print("\n真值标签文件格式示例:")
        print('  {"0": 1, "1": 0, "2": 1, ...}')
        
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
