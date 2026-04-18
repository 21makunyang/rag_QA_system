"""
RAG召回率评估模块

专门用于评估RAG系统中检索阶段的召回率性能。
支持多种召回率计算方式和阈值设置。
"""

import logging
import platform
from typing import List, Dict, Any, Tuple, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class RecallEvaluator:
    """RAG召回率评估器"""

    def __init__(self, similarity_threshold: float = 0.7, top_k: int = 5):
        """
        初始化召回率评估器

        Args:
            similarity_threshold: 语义相似度阈值
            top_k: 检索时返回的top k文档数量
        """
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.embedding_model = None
        self.embedding_tokenizer = None
        self._load_embedding_model()

    def _load_embedding_model(self):
        """加载嵌入模型用于语义相似度计算"""
        try:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.embedding_model = AutoModel.from_pretrained(model_name)

            if platform.system() == "Darwin" and torch.backends.mps.is_available():
                self.embedding_model = self.embedding_model.to("mps")
            elif torch.cuda.is_available():
                self.embedding_model = self.embedding_model.to("cuda")

            logger.info(f"Loaded embedding model: {model_name}")

        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")
            logger.warning("Semantic similarity calculations will use fallback methods")

    def calculate_recall_metrics(
        self,
        query: str,
        retrieved_contexts: List[str],
        ground_truth_contexts: List[str],
        retrieved_scores: List[float] = None,
        recall_direction: str = "gt_to_retrieved"
    ) -> Dict[str, float]:
        """
        计算召回率相关指标

        Args:
            query: 原始问题
            retrieved_contexts: 检索到的上下文列表
            ground_truth_contexts: 标准答案的参考上下文
            retrieved_scores: 检索结果的分数（可选）
            recall_direction: 召回率计算方向
                - "gt_to_retrieved": 以ground truth为中心（推荐）
                - "retrieved_to_gt": 以retrieved为中心

        Returns:
            召回率指标字典
        """
        metrics = {}

        # 1. 精确召回率 (基于关键词匹配)
        exact_recall = self._calculate_exact_recall(
            retrieved_contexts, ground_truth_contexts, direction=recall_direction
        )
        metrics["exact_recall"] = exact_recall

        # 2. 反向精确召回率（提供两种视角）
        reverse_recall = self._calculate_exact_recall(
            retrieved_contexts, ground_truth_contexts, direction="retrieved_to_gt"
        )
        metrics["reverse_exact_recall"] = reverse_recall

        # 2. 语义召回率 (基于语义相似度)
        semantic_recall = self._calculate_semantic_recall(
            query, retrieved_contexts, ground_truth_contexts
        )
        metrics["semantic_recall"] = semantic_recall

        # 3. Top-k召回率
        top_k_recall = self._calculate_top_k_recall(
            retrieved_contexts, ground_truth_contexts
        )
        metrics["top_k_recall"] = top_k_recall

        # 4. 加权召回率 (考虑检索分数)
        if retrieved_scores:
            weighted_recall = self._calculate_weighted_recall(
                retrieved_contexts, ground_truth_contexts, retrieved_scores
            )
            metrics["weighted_recall"] = weighted_recall

        # 5. 综合召回率分数
        metrics["overall_recall_score"] = self._calculate_overall_recall_score(metrics)

        return metrics

    def calculate_exact_recall(
        self,
        retrieved_contexts: List[str],
        ground_truth_contexts: List[str],
        direction: str = "gt_to_retrieved"
    ) -> float:
        """
        计算精确召回率（基于字符串包含关系）

        Args:
            retrieved_contexts: 检索到的上下文列表
            ground_truth_contexts: 标准答案的参考上下文
            direction: 计算方向
                - "gt_to_retrieved": 对每个ground truth，检查是否在retrieved中（推荐）
                - "retrieved_to_gt": 对每个retrieved，检查是否在ground truth中

        Returns:
            召回率
        """
        if not ground_truth_contexts or not retrieved_contexts:
            return 0.0

        if direction == "gt_to_retrieved":
            # 方法1：以ground truth为中心（推荐）
            # 对每个ground truth，检查是否在任意retrieved中
            recalled_count = 0
            for gt_context in ground_truth_contexts:
                for retrieved_context in retrieved_contexts:
                    # 如果标准上下文的关键词出现在检索结果中
                    gt_keywords = set(gt_context.lower().split())
                    retrieved_words = set(retrieved_context.lower().split())
                    overlap = len(gt_keywords.intersection(retrieved_words))

                    # 如果重叠度超过阈值，认为召回成功
                    overlap_ratio = overlap / len(gt_keywords) if gt_keywords else 0
                    if overlap_ratio >= self.similarity_threshold:
                        recalled_count += 1
                        break

            return recalled_count / len(ground_truth_contexts)

        else:
            # 方法2：以retrieved为中心（您建议的方式）
            # 对每个retrieved，检查是否在任意ground truth中
            relevant_retrieved = 0
            for retrieved_context in retrieved_contexts:
                for gt_context in ground_truth_contexts:
                    # 如果检索上下文的关键词出现在标准答案中
                    retrieved_keywords = set(retrieved_context.lower().split())
                    gt_words = set(gt_context.lower().split())
                    overlap = len(retrieved_keywords.intersection(gt_words))

                    # 如果重叠度超过阈值，认为相关
                    overlap_ratio = overlap / len(retrieved_keywords) if retrieved_keywords else 0
                    if overlap_ratio >= self.similarity_threshold:
                        relevant_retrieved += 1
                        break

            # 召回率 = 相关的retrieved / 所有的retrieved
            return relevant_retrieved / len(retrieved_contexts)

    def calculate_semantic_recall(
        self,
        question: str,
        retrieved_contexts: List[str],
        ground_truth_contexts: List[str]
    ) -> float:
        """计算语义召回率（基于语义相似度）"""
        if not self.embedding_model or not ground_truth_contexts:
            return self._calculate_exact_recall(retrieved_contexts, ground_truth_contexts)

        try:
            # 获取问题的嵌入
            question_embedding = self._get_embedding(question)

            # 获取所有上下文的嵌入
            gt_embeddings = [self._get_embedding(gt) for gt in ground_truth_contexts]
            retrieved_embeddings = [self._get_embedding(ret) for ret in retrieved_contexts]

            recalled_count = 0
            for gt_idx, gt_embedding in enumerate(gt_embeddings):
                # 计算与所有检索结果的相似度
                similarities = cosine_similarity([gt_embedding], retrieved_embeddings)[0]

                # 如果最大相似度超过阈值，认为召回成功
                max_similarity = np.max(similarities)
                if max_similarity >= self.similarity_threshold:
                    recalled_count += 1

            return recalled_count / len(ground_truth_contexts)

        except Exception as e:
            logger.error(f"Error in semantic recall calculation: {e}")
            return self._calculate_exact_recall(retrieved_contexts, ground_truth_contexts)

    def calculate_top_k_recall(
        self,
        retrieved_contexts: List[str],
        ground_truth_contexts: List[str]
    ) -> float:
        """计算Top-k召回率"""
        if not ground_truth_contexts or not retrieved_contexts:
            return 0.0

        # 只考虑top k个检索结果
        top_k_contexts = retrieved_contexts[:self.top_k]

        # 使用精确召回率计算方法
        return self._calculate_exact_recall(top_k_contexts, ground_truth_contexts)

    def calculate_weighted_recall(
        self,
        retrieved_contexts: List[str],
        ground_truth_contexts: List[str],
        retrieved_scores: List[float]
    ) -> float:
        """计算加权召回率（考虑检索置信度）"""
        if not ground_truth_contexts or not retrieved_scores:
            return self._calculate_exact_recall(retrieved_contexts, ground_truth_contexts)

        # 确保分数和上下文数量匹配
        if len(retrieved_scores) != len(retrieved_contexts):
            logger.warning("Scores and contexts length mismatch, using equal weights")
            retrieved_scores = [1.0] * len(retrieved_contexts)

        # 归一化分数
        scores = np.array(retrieved_scores)
        if scores.sum() > 0:
            weights = scores / scores.sum()
        else:
            weights = np.ones_like(scores) / len(scores)

        # 计算加权召回率
        total_weighted_recall = 0.0
        for gt_context in ground_truth_contexts:
            best_weight = 0.0
            for idx, retrieved_context in enumerate(retrieved_contexts):
                # 检查是否匹配
                gt_keywords = set(gt_context.lower().split())
                retrieved_words = set(retrieved_context.lower().split())
                overlap = len(gt_keywords.intersection(retrieved_words))
                overlap_ratio = overlap / len(gt_keywords) if gt_keywords else 0

                if overlap_ratio >= self.similarity_threshold:
                    best_weight = max(best_weight, weights[idx])

            total_weighted_recall += best_weight

        return total_weighted_recall / len(ground_truth_contexts)

    def calculate_overall_recall_score(self, metrics: Dict[str, float]) -> float:
        """计算综合召回率分数"""
        weights = {
            "exact_recall": 0.2,
            "semantic_recall": 0.4,
            "top_k_recall": 0.4
        }

        # 加权平均
        total_score = 0.0
        total_weight = 0.0

        for metric_name, weight in weights.items():
            if metric_name in metrics:
                total_score += metrics[metric_name] * weight
                total_weight += weight

        # 如果有加权召回率，给予额外权重
        if "weighted_recall" in metrics:
            total_score += metrics["weighted_recall"] * 0.1
            total_weight += 0.1

        return total_score / total_weight if total_weight > 0 else 0.0

    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的嵌入向量"""
        if not self.embedding_model or not self.embedding_tokenizer:
            # 返回随机向量作为fallback
            return np.random.randn(384)

        try:
            inputs = self.embedding_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu().numpy()

            return embedding

        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return np.random.randn(384)

    def evaluate_batch(
        self,
        questions: List[str],
        retrieved_contexts_list: List[List[str]],
        ground_truth_contexts_list: List[List[str]],
        retrieved_scores_list: List[List[float]] = None
    ) -> Dict[str, Any]:
        """
        批量评估召回率

        Args:
            questions: 问题列表
            retrieved_contexts_list: 每个问题检索到的上下文列表
            ground_truth_contexts_list: 每个问题的标准答案上下文列表
            retrieved_scores_list: 每个问题的检索分数列表（可选）

        Returns:
            批量评估结果
        """
        if retrieved_scores_list is None:
            retrieved_scores_list = [None] * len(questions)

        individual_results = []
        all_metrics = {
            "exact_recall": [],
            "semantic_recall": [],
            "top_k_recall": [],
            "weighted_recall": [],
            "overall_recall_score": []
        }

        for i, (question, retrieved_contexts, ground_truth_contexts, scores) in enumerate(
            zip(questions, retrieved_contexts_list, ground_truth_contexts_list, retrieved_scores_list)
        ):
            try:
                metrics = self.calculate_recall_metrics(
                    query=question,
                    retrieved_contexts=retrieved_contexts,
                    ground_truth_contexts=ground_truth_contexts,
                    retrieved_scores=scores
                )

                individual_results.append({
                    "question": question,
                    "metrics": metrics
                })

                # 收集所有指标
                for metric_name, value in metrics.items():
                    if metric_name in all_metrics:
                        all_metrics[metric_name].append(value)

            except Exception as e:
                logger.error(f"Error evaluating question {i}: {e}")
                individual_results.append({
                    "question": question,
                    "error": str(e)
                })

        # 计算汇总统计
        aggregate_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:
                aggregate_metrics[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values)
                }

        return {
            "individual_results": individual_results,
            "aggregate_metrics": aggregate_metrics,
            "total_questions": len(questions),
            "successful_evaluations": len([r for r in individual_results if "error" not in r])
        }

    def generate_recall_report(
        self,
        evaluation_data: Dict[str, Any],
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        生成召回率评估报告

        Args:
            evaluation_data: 评估数据
            output_path: 输出文件路径（可选）

        Returns:
            报告数据
        """
        questions = evaluation_data.get("question", [])
        answers = evaluation_data.get("answer", [])
        contexts = evaluation_data.get("contexts", [])
        ground_truth = evaluation_data.get("ground_truth", [])

        if not all([questions, answers, contexts, ground_truth]):
            raise ValueError("Missing required evaluation data fields")

        # 执行评估
        results = self.evaluate_batch(
            questions=questions,
            retrieved_contexts_list=contexts,
            ground_truth_contexts_list=[[gt] for gt in ground_truth]
        )

        # 生成报告
        report = {
            "evaluation_summary": {
                "total_questions": len(questions),
                "average_recall": results["aggregate_metrics"]["overall_recall_score"]["mean"],
                "recall_std": results["aggregate_metrics"]["overall_recall_score"]["std"],
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold
            },
            "detailed_metrics": results["aggregate_metrics"],
            "sample_evaluations": results["individual_results"][:5]  # 前5个样例
        }

        # 保存到文件
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Recall report saved to {output_path}")

        return report


def main():
    """测试召回率评估功能"""
    # 示例数据（使用用户提供的数据）
    data_samples = {
        'question': [
            'When was the first Super Bowl held?',
            'Which team has won the most Super Bowl championships?'
        ],
        'answer': [
            'The first Super Bowl was held on January 15, 1967',
            'The team that has won the most Super Bowls is the New England Patriots'
        ],
        'contexts': [
            ['The first AFL-NFL World Championship Game was an American football game held on January 15, 1967, at the Los Angeles Memorial Coliseum'],
            ['The Green Bay Packers... are based in Green Bay, Wisconsin.', 'The Packers compete in the National Football League...']
        ],
        'ground_truth': [
            'The first Super Bowl was held on January 15, 1967',
            'The New England Patriots have won a record six Super Bowl championships'
        ]
    }

    # 创建评估器
    evaluator = RecallEvaluator(similarity_threshold=0.7, top_k=5)

    # 生成报告
    report = evaluator.generate_recall_report(data_samples)

    # 打印结果
    print("RAG召回率评估报告")
    print("=" * 50)
    print(f"评估问题数量: {report['evaluation_summary']['total_questions']}")
    print(f"平均召回率: {report['evaluation_summary']['average_recall']:.3f}")
    print(f"召回率标准差: {report['evaluation_summary']['recall_std']:.3f}")
    print("\n详细指标:")
    for metric_name, stats in report['detailed_metrics'].items():
        print(f"\n{metric_name}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.3f}")

if __name__ == "__main__":
    main()