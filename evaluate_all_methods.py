#!/usr/bin/env python
"""
综合评估脚本：批量评估多个方法的结果并生成汇总表格
支持ROUGE, BLEU, BERTScore, 平均检索时间, 平均生成时间等指标
"""

import json
import sys
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# 尝试导入pandas，如果不可用则使用纯Python生成表格
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Will use simple table format. Install with: pip install pandas")

sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置HuggingFace镜像（如果环境变量未设置）
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ROUGE
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not available. Install with: pip install rouge-score")

# BLEU
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    BLEU_AVAILABLE = True
    try:
        import nltk
        import nltk.data
        # 检查punkt数据是否已存在
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            # 如果数据不存在，尝试下载
            try:
                nltk.download('punkt_tab', quiet=True)
            except:
                try:
                    nltk.download('punkt', quiet=True)
                except:
                    print("Warning: NLTK punkt数据下载失败，BLEU评估可能不可用")
                    print("  提示：可以手动下载: python -c \"import nltk; nltk.download('punkt')\"")
    except Exception:
        pass
except ImportError:
    BLEU_AVAILABLE = False
    print("Warning: nltk not available. Install with: pip install nltk")

# BERTScore
try:
    import bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("Warning: bert_score not available. Install with: pip install bert-score")


# ============================================================================
# 配置文件：请在这里填写要评估的文件列表和对应的名称
# ============================================================================
# 格式：[文件路径, 方法名称]
# 示例：
# [
#     ["./benchmark/results/medqa_baseline.json", "Baseline"],
#     ["./benchmark/results/medqa_cuckoo_depth1.json", "Depth=1"],
#     ["./benchmark/results/medqa_cuckoo_depth2.json", "Depth=2"],
#     ["./benchmark/results/medqa_cuckoo_depth3.json", "Depth=3"],
# ]

METHODS_CONFIG = [
    # 请在这里填写你的文件列表和对应的名称
    # ["./benchmark/results/medqa_baseline.json", "Baseline"],
    # ["./benchmark/results/medqa_cuckoo_depth1.json", "Depth=1"],
    # ["./benchmark/results/medqa_cuckoo_depth2.json", "Depth=2"],
    # ["./benchmark/results/medqa_cuckoo_depth3.json", "Depth=3"],
]


def extract_answer_from_text(text: str) -> str:
    """
    从文本中提取答案，支持两种格式：
    1. **Answer: xxx** (答案在**内，不跨行)
    2. **Answer:** xxx (答案在**外，单行，不跨行)
    返回提取的答案，如果未找到则返回原文本
    """
    if not text:
        return text
    
    # 格式1: **Answer: xxx** (答案在**内，不跨行)
    # 匹配到第一个**结束，不跨行
    pattern1 = r'\*\*Answer:\s*([^*\n]+)\*\*'
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        if answer:
            return answer
    
    # 格式2: **Answer:** xxx (答案在**外，单行，不跨行)
    # 匹配**Answer:**后面到换行符或行尾的内容（不跨行）
    pattern2 = r'\*\*Answer:\*\*\s*([^\n]+)'
    match = re.search(pattern2, text, re.IGNORECASE | re.MULTILINE)
    if match:
        answer = match.group(1).strip()
        # 移除可能的**标记（如果答案中有）
        answer = re.sub(r'\*\*', '', answer)
        # 移除末尾可能的句号后的内容（如果有其他标记）
        answer = re.sub(r'\.\s*\(.*?\)\s*$', '.', answer)  # 移除末尾的括号说明
        if answer:
            return answer
    
    # 如果都没找到，返回原文本
    return text


def evaluate_with_rouge(prediction: str, reference: str | list, scorer):
    """使用ROUGE指标评估"""
    if not ROUGE_AVAILABLE:
        return None
    
    # 兼容aa_lcr数据集：如果reference是列表，转换为字符串
    if isinstance(reference, list):
        if len(reference) > 0:
            # 如果列表中有多个答案，使用第一个（或合并所有答案）
            reference = " ".join(str(r) for r in reference if r)
        else:
            reference = ""
    
    # 确保prediction和reference都是字符串
    if not isinstance(prediction, str):
        prediction = str(prediction) if prediction else ""
    if not isinstance(reference, str):
        reference = str(reference) if reference else ""
    
    scores = scorer.score(reference, prediction)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure,
    }


def evaluate_with_bleu(prediction: str, reference: str | list):
    """使用BLEU指标评估"""
    if not BLEU_AVAILABLE:
        return None
    
    # 兼容aa_lcr数据集：如果reference是列表，转换为字符串
    if isinstance(reference, list):
        if len(reference) > 0:
            # 如果列表中有多个答案，使用第一个（或合并所有答案）
            reference = " ".join(str(r) for r in reference if r)
        else:
            reference = ""
    
    # 确保prediction和reference都是字符串
    if not isinstance(prediction, str):
        prediction = str(prediction) if prediction else ""
    if not isinstance(reference, str):
        reference = str(reference) if reference else ""
    
    try:
        # 检查punkt数据是否可用
        import nltk
        import nltk.data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            # punkt数据不可用，返回None
            return None
        
        # 使用word_tokenize进行分词
        pred_tokens = word_tokenize(prediction.lower())
        ref_tokens = word_tokenize(reference.lower())
        
        # 使用平滑函数避免0分
        smoothing_function = SmoothingFunction().method1
        bleu_score = sentence_bleu(
            [ref_tokens], 
            pred_tokens,
            smoothing_function=smoothing_function
        )
        
        return float(bleu_score)
    except LookupError:
        # punkt数据不存在
        return None
    except Exception as e:
        # 其他错误（如分词失败等）
        return None


def evaluate_with_bertscore(predictions: list, references: list, batch_size: int = 32):
    """使用BERTScore评估（批量处理）"""
    if not BERTSCORE_AVAILABLE:
        return None
    
    try:
        P, R, F1 = bert_score.score(
            predictions, 
            references, 
            lang='en',
            verbose=False,
            batch_size=batch_size,
            device='cpu'
        )
        return F1.tolist()
    except Exception as e:
        print(f"BERTScore计算错误: {e}")
        return None


def evaluate_single_file(results_file: str, method_name: str, skip_bertscore: bool = False) -> Dict[str, Any]:
    """评估单个结果文件，返回所有指标"""
    
    if not os.path.exists(results_file):
        print(f"⚠ 文件不存在: {results_file}")
        return None
    
    print(f"\n正在评估: {method_name}")
    print(f"  文件: {results_file}")
    
    # 读取结果
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"  ✗ 读取文件失败: {e}")
        return None
    
    if not results or len(results) == 0:
        print(f"  ⚠ 文件为空")
        return None
    
    # 初始化ROUGE scorer
    rouge_scorer_obj = None
    if ROUGE_AVAILABLE:
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # 收集所有predictions和references
    predictions = []
    references = []
    retrieval_times = []
    generation_times = []
    total_times = []
    
    for result in results:
        prediction_raw = result.get("answer", "")
        reference_raw = result.get("expected_answer", "")
        
        # 兼容aa_lcr数据集：如果expected_answer是列表，转换为字符串
        if isinstance(reference_raw, list):
            if len(reference_raw) > 0:
                # 如果列表中有多个答案，合并所有答案（用空格分隔）
                reference = " ".join(str(r) for r in reference_raw if r)
            else:
                reference = ""
        else:
            reference = str(reference_raw) if reference_raw else ""
        
        if not prediction_raw or not reference:
            continue
        
        # 从answer中提取实际答案（格式：**Answer: xxx**）
        prediction = extract_answer_from_text(prediction_raw)
        
        predictions.append(prediction)
        references.append(reference)
        
        # 收集时间信息
        if "retrieval_time" in result and result["retrieval_time"] is not None:
            retrieval_times.append(result["retrieval_time"])
        if "generation_time" in result and result["generation_time"] is not None:
            generation_times.append(result["generation_time"])
        if "time" in result and result["time"] is not None:
            total_times.append(result["time"])
    
    if len(predictions) == 0:
        print(f"  ⚠ 没有有效的预测-参考对")
        return None
    
    print(f"  有效样本数: {len(predictions)}")
    
    # ROUGE评估
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    # BLEU评估
    bleu_scores = []
    
    # 逐个样本评估ROUGE和BLEU
    for pred, ref in zip(predictions, references):
        # ROUGE
        if rouge_scorer_obj:
            rouge_scores = evaluate_with_rouge(pred, ref, rouge_scorer_obj)
            if rouge_scores:
                rouge1_scores.append(rouge_scores['rouge1'])
                rouge2_scores.append(rouge_scores['rouge2'])
                rougeL_scores.append(rouge_scores['rougeL'])
        
        # BLEU
        bleu_score = evaluate_with_bleu(pred, ref)
        if bleu_score is not None:
            bleu_scores.append(bleu_score)
    
    # BERTScore批量评估
    bertscore_scores = None
    if BERTSCORE_AVAILABLE and len(predictions) > 0 and not skip_bertscore:
        print(f"  正在计算BERTScore...")
        bertscore_scores = evaluate_with_bertscore(
            predictions, 
            references, 
            batch_size=min(32, len(predictions))
        )
    
    # 计算平均值
    metrics = {
        "method": method_name,
        "file": results_file,
        "samples": len(predictions),
    }
    
    # ROUGE指标
    if rouge1_scores:
        metrics["ROUGE-1"] = sum(rouge1_scores) / len(rouge1_scores)
        metrics["ROUGE-2"] = sum(rouge2_scores) / len(rouge2_scores)
        metrics["ROUGE-L"] = sum(rougeL_scores) / len(rougeL_scores)
    else:
        metrics["ROUGE-1"] = None
        metrics["ROUGE-2"] = None
        metrics["ROUGE-L"] = None
    
    # BLEU指标
    if bleu_scores:
        metrics["BLEU"] = sum(bleu_scores) / len(bleu_scores)
    else:
        metrics["BLEU"] = None
    
    # BERTScore指标
    if bertscore_scores:
        metrics["BERTScore"] = sum(bertscore_scores) / len(bertscore_scores)
    else:
        metrics["BERTScore"] = None
    
    # 时间指标
    if retrieval_times:
        metrics["平均检索时间(秒)"] = sum(retrieval_times) / len(retrieval_times)
    else:
        metrics["平均检索时间(秒)"] = None
    
    if generation_times:
        metrics["平均生成时间(秒)"] = sum(generation_times) / len(generation_times)
    else:
        metrics["平均生成时间(秒)"] = None
    
    if total_times:
        metrics["平均总时间(秒)"] = sum(total_times) / len(total_times)
    else:
        metrics["平均总时间(秒)"] = None
    
    print(f"  ✓ 评估完成")
    
    return metrics


def print_aligned_table(formatted_data: List[Dict[str, Any]], display_columns: List[str]):
    """打印对齐的表格（不依赖pandas）"""
    # 计算每列的最大宽度
    col_widths = {}
    for col in ['方法'] + display_columns:
        # 计算列名宽度
        header_width = len(str(col))
        # 计算该列所有数据的最大宽度
        data_widths = [len(str(row.get(col, 'N/A'))) for row in formatted_data]
        col_widths[col] = max(header_width, max(data_widths) if data_widths else 0)
        # 至少保留最小宽度
        if col == '方法':
            col_widths[col] = max(col_widths[col], 8)
        else:
            col_widths[col] = max(col_widths[col], len(col))
    
    # 打印表头
    header_parts = []
    for col in ['方法'] + display_columns:
        header_parts.append(str(col).ljust(col_widths[col]))
    header = " | ".join(header_parts)
    print(header)
    
    # 打印分隔线
    separator_parts = []
    for col in ['方法'] + display_columns:
        separator_parts.append("-" * col_widths[col])
    separator = "-+-".join(separator_parts)
    print(separator)
    
    # 打印数据行
    for row in formatted_data:
        row_parts = []
        for col in ['方法'] + display_columns:
            value = str(row.get(col, 'N/A'))
            row_parts.append(value.ljust(col_widths[col]))
        data_row = " | ".join(row_parts)
        print(data_row)


def generate_table(all_metrics: List[Dict[str, Any]], output_file: Optional[str] = None):
    """生成汇总表格"""
    
    if not all_metrics:
        print("没有可用的评估结果")
        return
    
    # 定义列的顺序
    column_order = [
        'samples',
        '平均检索时间(秒)',
        '平均生成时间(秒)',
        '平均总时间(秒)',
        'ROUGE-1',
        'ROUGE-2',
        'ROUGE-L',
        'BLEU',
        'BERTScore'
    ]
    
    # 收集所有列名
    all_columns = set()
    for metrics in all_metrics:
        all_columns.update(metrics.keys())
    
    # 移除不需要显示的列
    all_columns.discard('file')
    all_columns.discard('method')
    
    # 重新排列列的顺序
    display_columns = [col for col in column_order if col in all_columns]
    display_columns.extend([col for col in sorted(all_columns) if col not in column_order])
    
    # 格式化数据
    formatted_data = []
    for metrics in all_metrics:
        row = {'方法': metrics.get('method', 'N/A')}
        for col in display_columns:
            value = metrics.get(col)
            if value is None:
                row[col] = "N/A"
            elif isinstance(value, (int, float)):
                if isinstance(value, float):
                    row[col] = f"{value:.4f}"
                else:
                    row[col] = str(value)
            else:
                row[col] = str(value)
        formatted_data.append(row)
    
    # 打印表格（使用自定义对齐函数）
    print("\n" + "=" * 100)
    print("评估结果汇总表格")
    print("=" * 100)
    print()
    print_aligned_table(formatted_data, display_columns)
    print()
    print("=" * 100)
    
    # 保存文件
    if output_file:
        # 保存为CSV
        csv_file = output_file.replace('.json', '.csv') if output_file.endswith('.json') else output_file + '.csv'
        with open(csv_file, 'w', encoding='utf-8-sig') as f:
            # 写入表头
            f.write(",".join(['方法'] + display_columns) + "\n")
            # 写入数据
            for row in formatted_data:
                f.write(",".join(str(row.get(col, 'N/A')) for col in ['方法'] + display_columns) + "\n")
        print(f"✓ 表格已保存到: {csv_file}")
        
        # 如果pandas可用，生成Markdown表格
        if PANDAS_AVAILABLE:
            try:
                df = pd.DataFrame(formatted_data)
                df.set_index('方法', inplace=True)
                df_display = df[display_columns]
                markdown_file = output_file.replace('.json', '.md') if output_file.endswith('.json') else output_file + '.md'
                with open(markdown_file, 'w', encoding='utf-8') as f:
                    f.write("# 评估结果汇总\n\n")
                    f.write(df_display.to_markdown())
                print(f"✓ Markdown表格已保存到: {markdown_file}")
            except:
                pass  # 如果to_markdown不可用，跳过
    
    # 保存为JSON（包含完整信息）
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        print(f"✓ 详细结果已保存到: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="批量评估多个方法的结果并生成汇总表格",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  1. 编辑脚本中的 METHODS_CONFIG 列表，填写要评估的文件和方法名称
  2. 运行: python benchmark/evaluate_all_methods.py --output results_summary.json
  
或者使用命令行参数:
  python benchmark/evaluate_all_methods.py \\
    --files file1.json file2.json file3.json \\
    --names "Method 1" "Method 2" "Method 3" \\
    --output results_summary.json
        """
    )
    parser.add_argument('--files', type=str, nargs='+', default=None,
                       help='要评估的结果文件列表（覆盖配置文件）')
    parser.add_argument('--names', type=str, nargs='+', default=None,
                       help='对应的方法名称列表（必须与--files数量相同）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出汇总结果文件路径（JSON格式）')
    parser.add_argument('--skip-bertscore', action='store_true',
                       help='跳过BERTScore评估（BERTScore首次运行需要下载模型，较慢）')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（JSON格式，包含files和names列表）')
    
    args = parser.parse_args()
    
    # 确定要评估的文件列表
    methods_to_evaluate = []
    
    if args.config:
        # 从配置文件读取
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
            methods_list = config.get('methods', [])
            # 转换为 (file, name) 元组列表
            methods_to_evaluate = [(item['file'], item['name']) for item in methods_list]
        except Exception as e:
            print(f"✗ 读取配置文件失败: {e}")
            return
    elif args.files and args.names:
        # 从命令行参数读取
        if len(args.files) != len(args.names):
            print("✗ 错误: --files 和 --names 的数量必须相同")
            return
        methods_to_evaluate = list(zip(args.files, args.names))
    elif METHODS_CONFIG:
        # 使用脚本中的配置
        methods_to_evaluate = METHODS_CONFIG
    else:
        print("✗ 错误: 请提供要评估的文件列表")
        print("  方法1: 编辑脚本中的 METHODS_CONFIG 列表")
        print("  方法2: 使用 --files 和 --names 参数")
        print("  方法3: 使用 --config 指定配置文件")
        return
    
    if not methods_to_evaluate:
        print("✗ 错误: 没有找到要评估的文件")
        return
    
    print("=" * 100)
    print("批量评估多个方法")
    print("=" * 100)
    print(f"\n支持的指标:")
    print(f"  ROUGE: {'✓' if ROUGE_AVAILABLE else '✗'}")
    print(f"  BLEU: {'✓' if BLEU_AVAILABLE else '✗'}")
    print(f"  BERTScore: {'✓' if BERTSCORE_AVAILABLE else '✗'}")
    print(f"\n要评估的方法数量: {len(methods_to_evaluate)}")
    print("=" * 100)
    
    # 评估所有方法
    all_metrics = []
    
    for file_path, method_name in methods_to_evaluate:
        metrics = evaluate_single_file(file_path, method_name, skip_bertscore=args.skip_bertscore)
        if metrics:
            all_metrics.append(metrics)
    
    if not all_metrics:
        print("\n✗ 没有成功评估任何方法")
        return
    
    # 生成表格
    output_file = args.output or "./benchmark/results/evaluation_summary.json"
    generate_table(all_metrics, output_file)
    
    print("\n" + "=" * 100)
    print("评估完成！")
    print("=" * 100)


if __name__ == "__main__":
    main()

