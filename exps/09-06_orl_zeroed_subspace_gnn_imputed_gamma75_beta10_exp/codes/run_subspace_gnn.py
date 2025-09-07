import os
import sys
import argparse
import numpy as np
from pathlib import Path
from subspace_aware_gnn import process_mat_file

def main():
    parser = argparse.ArgumentParser(description='运行子空间感知的图神经网络进行缺失值填补')
    # 必需参数
    parser.add_argument('--input', type=str, required=True, 
                        help='输入.mat文件路径')

    # 可选参数
    parser.add_argument('--output_dir', type=str, default='data/datasets', 
                        help='输出目录，默认为data/datasets')
    parser.add_argument('--treat_zeros', action='store_true', 
                        help='将0值视为缺失值')
    parser.add_argument('--k', type=int, default=10, 
                        help='KNN图中的近邻数量')
    parser.add_argument('--epochs', type=int, default=200, 
                        help='训练轮数')
    parser.add_argument('--alpha', type=float, default=0.5, 
                        help='子空间内连接的权重系数 (0-1)，控制特征相似度(alpha)与标签相似度(1-alpha)的平衡。'
                             '较大的值(如0.8)更注重特征相似性，较小的值(如0.5)更注重标签/子空间结构。'
                             '对于COIL100等子空间结构明显的数据集，可以考虑使用较小的值如0.5-0.7。')
    parser.add_argument('--auto_cluster', action='store_true',
                        help='填补完成后自动运行子空间聚类')
    parser.add_argument('--list_datasets', action='store_true',
                        help='列出data/datasets目录下所有可用的.mat文件')

    args = parser.parse_args()

    # 列出可用数据集
    if args.list_datasets:
        print("\n可用的数据集文件:")
        dataset_dir = Path("data/datasets")
        if dataset_dir.exists():
            for file_path in dataset_dir.glob("**/*.mat"):
                print(f"  {file_path}")
        else:
            print(f"  目录 {dataset_dir} 不存在")
        return

    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 文件 '{args.input}' 不存在")
        print("\n使用示例:")
        print("  # 仅数据填补:")
        print("  python run_subspace_gnn.py --input data/datasets/COIL100_random_zero.mat --treat_zeros")
        print("  # 数据填补 + 自动子空间聚类:")
        print("  python run_subspace_gnn.py --input data/datasets/COIL100_random_zero.mat --treat_zeros --auto_cluster")
        print("  # 高级参数设置:")
        print("  python run_subspace_gnn.py --input data/datasets/ORL_32x32_random_zero.mat --k 15 --epochs 300 --alpha 0.7 --auto_cluster")
        print("  # 列出可用数据集:")
        print("  python run_subspace_gnn.py --list_datasets")
        return

    # 获取输入文件的基本名称
    input_filename = os.path.basename(args.input)

    # 添加对特定数据集的检查
    if input_filename in ["ORL_32x32.mat", "COIL100.mat"]:
        print(f"提示: 数据集 '{input_filename}' 被认为是原始数据集，没有缺失数据，不必进行填补。")
        print("请提供带有缺失值的版本，例如 'COIL100_zeroed_20.mat' 或 'ORL_32x32_zeroed_30.mat'。")
        return # 退出程序，不执行后续填补操作

    # 处理数据
    try:
        output_path = process_mat_file(
            args.input, 
            args.output_dir, 
            treat_zeros_as_missing=args.treat_zeros,
            k=args.k,
            epochs=args.epochs,
            alpha=args.alpha
        )
        
        print("\n✅ 数据填补完成!")
        print(f"📂 输出文件: {output_path}")

        # 如果启用自动聚类，则运行子空间聚类
        if args.auto_cluster:
            print("\n" + "="*60)
            print("🚀 开始自动运行子空间聚类...")
            print("="*60)

            # 检查main_subspace.py是否存在
            subspace_script = "subspace_cluster.py"
            if not os.path.exists(subspace_script):
                print(f"❌ 错误: 找不到子空间聚类脚本 '{subspace_script}'")
                print("请确保subspace_cluster.py文件在当前目录中")
                return

            # 运行子空间聚类
            try:
                import subprocess

                print(f"📂 使用填补后的数据文件: {output_path}")

                # 根据输出文件名智能确定数据集类型
                filename = os.path.basename(output_path).lower()
                dataset_name = None

                # 尝试匹配COIL100
                if 'coil100' in filename:
                    if '_zeroed_20' in filename and '_subspace_gnn_imputed' in filename:
                        dataset_name = 'coil100_zeroed_20_subspace_gnn_imputed'
                    elif '_zeroed_30' in filename and '_subspace_gnn_imputed' in filename:
                        dataset_name = 'coil100_zeroed_30_subspace_gnn_imputed'
                    elif '_zeroed_40' in filename and '_subspace_gnn_imputed' in filename:
                        dataset_name = 'coil100_zeroed_40_subspace_gnn_imputed'
                    elif '_zeroed_50' in filename and '_subspace_gnn_imputed' in filename:
                        dataset_name = 'coil100_zeroed_50_subspace_gnn_imputed'
                    elif '_zeroed' in filename and '_subspace_gnn_imputed' in filename: # 匹配没有具体百分比的
                        dataset_name = 'coil100_zeroed_subspace_gnn_imputed'
                    else: # Fallback for COIL100 if a specific pattern isn't matched
                        dataset_name = 'coil100_zeroed_gnn_imputed' 
                # 尝试匹配ORL
                elif 'orl' in filename:
                    if '_zeroed_20' in filename and '_subspace_gnn_imputed' in filename:
                        dataset_name = 'orl_zeroed_20_subspace_gnn_imputed'
                    elif '_zeroed_30' in filename and '_subspace_gnn_imputed' in filename:
                        dataset_name = 'orl_zeroed_30_subspace_gnn_imputed'
                    elif '_zeroed_40' in filename and '_subspace_gnn_imputed' in filename:
                        dataset_name = 'orl_zeroed_40_subspace_gnn_imputed'
                    elif '_zeroed_50' in filename and '_subspace_gnn_imputed' in filename:
                        dataset_name = 'orl_zeroed_50_subspace_gnn_imputed'
                    elif '_zeroed' in filename and '_subspace_gnn_imputed' in filename: # 匹配没有具体百分比的
                        dataset_name = 'orl_zeroed_subspace_gnn_imputed'
                    else: # Fallback for ORL if a specific pattern isn't matched
                        dataset_name = 'orl_zeroed_gnn_imputed'
                
                if dataset_name is None:
                    # 最终的默认值，以防上述所有匹配都失败
                    dataset_name = 'coil100_random_zero_gnn_imputed'
                    print(f"⚠️ 无法从文件名 '{filename}' 推断出精确的数据集类型，使用默认: {dataset_name}")

                print(f"🎯 推断数据集类型: {dataset_name}")

                # 构建命令 - main_subspace.py使用--data参数
                cmd = [sys.executable, subspace_script, "--data", dataset_name]

                print(f"🔧 执行命令: {' '.join(cmd)}")
                print("-" * 60)

                # 运行子空间聚类脚本
                result = subprocess.run(cmd, capture_output=False, text=True)

                if result.returncode == 0:
                    print("-" * 60)
                    print("✅ 子空间聚类完成！")
                    print("🎯 完整流程：数据填补 → 子空间聚类 已成功完成")
                else:
                    print("-" * 60)
                    print(f"❌ 子空间聚类执行失败，返回码: {result.returncode}")
                    print("💡 请检查subspace_cluster.py是否支持该数据集类型")

            except Exception as e:
                print(f"❌ 运行子空间聚类时出错: {e}")
                print("💡 您可以手动运行以下命令:")
                # 这里可以根据推断出的dataset_name提供手动运行的建议
                if dataset_name:
                    print(f"   python subspace_cluster.py --data {dataset_name}")
                else:
                    print(f"   无法自动生成手动运行命令，请根据填补后的文件 '{output_path}' 手动调整数据集参数。")
                
        else:
            print("\n💡 提示: 如需自动运行子空间聚类，请添加 --auto_cluster 参数")
            print(f"   python run_subspace_gnn.py --input {args.input} --auto_cluster")
            print("   完整流程示例:")
            print(f"   python run_subspace_gnn.py --input {args.input} --treat_zeros --auto_cluster")

        print("\n📊 您也可以使用以下命令查看填补结果:")
        print(f"  python view_mat_file_simple.py --file {output_path}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()