"""Dify平台适配器"""

import json
import time
from typing import Any

from agent_evaluator.adapters.base import AdapterResponse, PerformanceMetrics, PlatformAdapter
from agent_evaluator.adapters.streaming import StreamingAccumulator
from agent_evaluator.utils.logger import get_logger

logger = get_logger(__name__)


class DifyAdapter(PlatformAdapter):
    """Dify平台适配器"""

    def __init__(self, api_config: dict[str, Any], show_streaming_content: bool = False):
        """
        初始化Dify适配器
        
        Args:
            api_config: API配置字典
            show_streaming_content: 是否显示流式输出的详细内容
        """
        super().__init__(api_config)
        self.show_streaming_content = show_streaming_content

    def _get_headers(self) -> dict[str, str]:
        """获取Dify API请求头"""
        headers = super()._get_headers()
        api_key = self.api_config.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def invoke(
        self,
        input: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> AdapterResponse:
        """
        调用Dify API

        Args:
            input: 用户输入（如果inputs中包含query，则此参数可能被忽略）
            stream: 是否使用流式输出
            **kwargs: 其他参数，包括：
                - conversation_id: 对话ID（可选）
                - user: 用户ID（可选）
                - app_id: 应用ID（可选，某些API可能需要）
                - inputs: 工作流输入变量字典（可选），如果包含query字段，则使用inputs.query作为实际查询

        Returns:
            AdapterResponse对象
        """
        logger.debug(f"调用Dify API，流式模式: {stream}, 输入长度: {len(input)}")
        if stream:
            return await self._invoke_streaming(input, **kwargs)
        else:
            return await self._invoke_non_streaming(input, **kwargs)

    async def _invoke_non_streaming(
        self,
        input: str,
        **kwargs: Any,
    ) -> AdapterResponse:
        """非流式调用"""
        if not self._client:
            raise RuntimeError("Adapter must be used as async context manager")

        base_url = self.api_config.get("base_url", "https://api.dify.ai/v1")
        app_id = kwargs.get("app_id") or self.api_config.get("app_id")
        # app_id 不是必需的，某些API可能不需要
        # if not app_id:
        #     raise ValueError("app_id is required")

        url = f"{base_url}/chat-messages"
        
        # 支持通过inputs传递工作流输入变量
        inputs = kwargs.get("inputs", {})
        # 如果inputs中包含query，使用它；否则使用顶层的query
        if "query" in inputs:
            # 如果inputs.query是实际的问题，则使用它作为query
            payload = {
                "inputs": inputs,
                "query": "-",  # 占位符，实际查询在inputs.query中
                "response_mode": "blocking",
                "conversation_id": kwargs.get("conversation_id"),
                "user": kwargs.get("user", "agent-evaluator"),
            }
        else:
            payload = {
                "inputs": inputs,
                "query": input,
                "response_mode": "blocking",  # 非流式模式
                "conversation_id": kwargs.get("conversation_id"),
                "user": kwargs.get("user", "agent-evaluator"),
            }

        logger.debug(f"发送非流式请求到: {url}")
        start_time = time.time()
        response = await self._client.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        total_time = time.time() - start_time
        logger.debug(f"API调用完成，耗时: {total_time:.3f}秒")
        
        # 诊断：记录API响应的关键字段
        logger.debug(f"Dify API响应字段: {list(data.keys())}")
        if "retrieved_contexts" in data:
            contexts_count = len(data.get("retrieved_contexts", []))
            logger.debug(f"API响应中包含retrieved_contexts，数量: {contexts_count}")
        else:
            logger.warning("API响应中未找到retrieved_contexts字段，可能是应用未配置RAG/知识库检索")
            # 检查是否有其他可能的字段名
            possible_fields = [k for k in data.keys() if "context" in k.lower() or "retriev" in k.lower()]
            if possible_fields:
                logger.debug(f"发现可能的上下文相关字段: {possible_fields}")

        # 解析响应
        answer = data.get("answer", "")
        contexts = data.get("retrieved_contexts", [])
        metadata = {
            "message_id": data.get("message_id"),
            "conversation_id": data.get("conversation_id"),
            "created_at": data.get("created_at"),
            **data,
        }

        # 提取性能指标
        performance = None
        if "metadata" in data:
            meta = data["metadata"]
            performance = PerformanceMetrics(
                total_time=total_time,
                total_tokens=meta.get("usage", {}).get("total_tokens", 0),
                input_tokens=meta.get("usage", {}).get("prompt_tokens", 0),
                output_tokens=meta.get("usage", {}).get("completion_tokens", 0),
            )
            logger.debug(f"性能指标: tokens={performance.total_tokens}, 输入={performance.input_tokens}, 输出={performance.output_tokens}")

        return AdapterResponse(
            answer=answer,
            contexts=contexts,
            metadata=metadata,
            performance=performance,
        )

    async def _invoke_streaming(
        self,
        input: str,
        **kwargs: Any,
    ) -> AdapterResponse:
        """流式调用"""
        if not self._client:
            raise RuntimeError("Adapter must be used as async context manager")

        base_url = self.api_config.get("base_url", "https://api.dify.ai/v1")
        app_id = kwargs.get("app_id") or self.api_config.get("app_id")
        # app_id 不是必需的，某些API可能不需要
        # if not app_id:
        #     raise ValueError("app_id is required")

        url = f"{base_url}/chat-messages"
        
        # 支持通过inputs传递工作流输入变量
        inputs = kwargs.get("inputs", {})
        # 如果inputs中包含query，使用它；否则使用顶层的query
        if "query" in inputs:
            # 如果inputs.query是实际的问题，则使用它作为query
            payload = {
                "inputs": inputs,
                "query": "-",  # 占位符，实际查询在inputs.query中
                "response_mode": "streaming",  # 流式模式
                "conversation_id": kwargs.get("conversation_id"),
                "user": kwargs.get("user", "agent-evaluator"),
            }
        else:
            payload = {
                "inputs": inputs,
                "query": input,
                "response_mode": "streaming",  # 流式模式
                "conversation_id": kwargs.get("conversation_id"),
                "user": kwargs.get("user", "agent-evaluator"),
            }

        logger.debug(f"发送流式请求到: {url}")
        start_time = time.time()
        accumulator = StreamingAccumulator()

        async with self._client.stream("POST", url, json=payload) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                # 解析SSE格式：data: {...}
                if line.startswith("data: "):
                    try:
                        event_data = line[6:]  # 移除 "data: " 前缀
                        if event_data == "[DONE]":
                            break

                        event = json.loads(event_data)
                        current_time = time.time() - start_time
                        
                        # 如果启用详细日志，记录每个事件
                        if self.show_streaming_content:
                            event_type = event.get("event", "unknown")
                            logger.debug(f"[流式事件] 类型: {event_type}, 时间: {current_time:.3f}s")
                            if event_type == "message":
                                answer_chunk = event.get("answer", "")
                                if answer_chunk:
                                    logger.debug(f"[流式内容] 答案片段: {answer_chunk[:100]}{'...' if len(answer_chunk) > 100 else ''}")
                            elif event_type == "node_finished":
                                data = event.get("data", {})
                                node_id = data.get("node_id", "unknown")
                                outputs = data.get("outputs", {})
                                logger.debug(f"[流式节点] 节点ID: {node_id}, 输出字段: {list(outputs.keys())}")
                                if "retrieved_contexts" in outputs:
                                    contexts = outputs["retrieved_contexts"]
                                    logger.debug(f"[流式上下文] 提取到 {len(contexts) if isinstance(contexts, list) else 1} 个上下文")
                            elif event_type == "workflow_finished":
                                data = event.get("data", {})
                                logger.debug(f"[流式完成] 总tokens: {data.get('total_tokens', 0)}, 耗时: {data.get('elapsed_time', 0):.3f}s")
                        
                        accumulator.accumulate(event, current_time)

                    except json.JSONDecodeError:
                        # 忽略无法解析的行
                        logger.debug(f"无法解析SSE行: {line[:50]}...")
                        continue

        # 流式完成后，构建完整响应
        total_time = time.time() - start_time
        logger.debug(f"流式API调用完成，耗时: {total_time:.3f}秒, 收到tokens: {len(accumulator.token_timestamps)}")
        
        # 诊断：记录提取到的contexts信息
        if accumulator.contexts:
            logger.debug(f"流式模式成功提取到 {len(accumulator.contexts)} 个retrieved_contexts")
        else:
            logger.warning("流式模式retrieved_contexts为空，可能是节点输出中未包含retrieved_contexts字段")
            # 尝试从metadata中查找
            if "nodes" in accumulator.metadata:
                nodes = accumulator.metadata.get("nodes", [])
                logger.debug(f"流式响应包含 {len(nodes)} 个节点，检查节点输出...")
                for i, node in enumerate(nodes):
                    outputs = node.get("outputs", {})
                    logger.debug(f"节点 {i} 的输出字段: {list(outputs.keys())}")
        
        return accumulator.to_adapter_response(start_time)
