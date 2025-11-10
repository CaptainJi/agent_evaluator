"""指标注册表（简化配置）"""

from typing import Any

from ragas.metrics import (
    AnswerAccuracy,
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    ResponseRelevancy,
)


def create_metric(
    metric_name: str, 
    llm: Any, 
    embeddings: Any | None = None,
    strictness: int | None = None,
) -> Any:
    """
    根据指标名称创建Ragas指标对象

    Args:
        metric_name: 指标名称（如 "faithfulness", "relevancy"等）
        llm: 评估用的LLM对象
        embeddings: 评估用的embeddings对象（可选）
        strictness: ResponseRelevancy的strictness参数（生成问题的数量，默认3，建议2以减少超时）

    Returns:
        Ragas指标对象

    Raises:
        ValueError: 不支持的指标名称
    """
    metric_name_lower = metric_name.lower().strip()

    # 映射指标名称到Ragas类
    metric_map: dict[str, type] = {
        "faithfulness": Faithfulness,
        "answer_relevancy": ResponseRelevancy,
        "response_relevancy": ResponseRelevancy,
        "relevancy": ResponseRelevancy,
        "context_precision": ContextPrecision,
        "context_recall": ContextRecall,
        "answer_correctness": AnswerCorrectness,
        "answer_accuracy": AnswerAccuracy,
    }

    metric_class = metric_map.get(metric_name_lower)
    if not metric_class:
        raise ValueError(
            f"不支持的指标: {metric_name}。支持的指标: {', '.join(metric_map.keys())}"
        )

    # 创建指标对象
    # 某些指标需要embeddings，某些只需要llm
    if metric_name_lower in ["answer_relevancy", "response_relevancy", "relevancy"]:
        if embeddings is None:
            raise ValueError(f"指标 {metric_name} 需要 embeddings 参数")
        # ResponseRelevancy支持strictness参数，默认3，但会导致多次LLM调用
        # 如果指定了strictness，使用它；否则使用默认值2（减少超时风险）
        if strictness is not None:
            metric_instance = metric_class(llm=llm, embeddings=embeddings, strictness=strictness)
        else:
            # 默认使用2而不是3，以减少LLM调用次数和超时风险
            # ResponseRelevancy会调用strictness次LLM来生成问题，每次调用可能需要30-40秒
            # strictness=2意味着2次调用，约60-80秒；strictness=3意味着3次调用，约90-120秒
            metric_instance = metric_class(llm=llm, embeddings=embeddings, strictness=2)
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"已创建 {metric_name} 指标，strictness={metric_instance.strictness if hasattr(metric_instance, 'strictness') else 'N/A'}")
        
        return metric_instance
    else:
        return metric_class(llm=llm)


def create_metrics(
    metric_names: list[str], llm: Any, embeddings: Any | None = None
) -> list[Any]:
    """
    批量创建指标对象

    Args:
        metric_names: 指标名称列表
        llm: 评估用的LLM对象
        embeddings: 评估用的embeddings对象（可选）

    Returns:
        Ragas指标对象列表
    """
    return [create_metric(name, llm, embeddings) for name in metric_names]
