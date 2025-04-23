from typing import Any, List, Optional
from langchain_core.embeddings import Embeddings
from zhipuai import ZhipuAI
from dotenv import load_dotenv
import os
import logging
import time
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class ZhipuAIEmbeddings(Embeddings):
    """智谱AI Embeddings包装器。
    
    使用智谱AI的embedding-3模型生成文本嵌入。
    
    Attributes:
        model: 使用的模型名称
        api_key: 智谱AI API密钥
        dimensions: 输出嵌入向量的维度（仅支持embedding-3及以上版本）
        client: 智谱AI客户端实例
        chunk_size: 文本分块大小（字符数）
        request_timeout: API请求超时时间（秒）
        batch_size: 批处理大小
    """
    
    def __init__(
        self,
        model: str = "embedding-3",
        api_key: Optional[str] = None,
        dimensions: int = 2048,  # 默认使用2048维度
        chunk_size: int = 2000,  # 默认分块大小
        request_timeout: int = 30,  # 默认30秒超时
        batch_size: int = 64,  # 批处理大小（智谱AI API限制最大64条）
        **kwargs: Any,
    ) -> None:
        """初始化ZhipuAI Embeddings。
        
        Args:
            model: 模型名称，默认为"embedding-3"
            api_key: 智谱AI API密钥，如果为None则从环境变量获取
            dimensions: 输出嵌入向量的维度，默认2048
            chunk_size: 文本分块大小（字符数），默认2000
            request_timeout: API请求超时时间（秒），默认30
            batch_size: 批处理大小，默认64（智谱AI API限制）
            **kwargs: 额外的关键字参数
        """
        self.model = model
        self.dimensions = dimensions
        self.chunk_size = chunk_size
        self.request_timeout = request_timeout
        self.batch_size = batch_size
        self.api_key = api_key or os.getenv("ZHIPUAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "智谱AI API密钥未设置。请通过参数传入或设置ZHIPUAI_API_KEY环境变量。"
            )
        self.client = ZhipuAI(api_key=self.api_key)
        logger.info(f"初始化完成：模型={model}, 维度={dimensions}, 批处理大小={batch_size}")

    def _split_text(self, text: str) -> List[str]:
        """将文本分割成较小的块。
        
        Args:
            text: 要分割的文本
            
        Returns:
            文本块列表
        """
        chunks = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        logger.info(f"文本已分割为 {len(chunks)} 个块")
        return chunks

    def _get_embedding_with_retry(self, text: str, max_retries: int = 3) -> List[float]:
        """带重试的API调用。
        
        Args:
            text: 要生成嵌入的文本
            max_retries: 最大重试次数
            
        Returns:
            嵌入向量
        """
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                    dimensions=self.dimensions  # 始终指定维度
                )
                elapsed_time = time.time() - start_time
                logger.debug(f"API调用耗时: {elapsed_time:.2f}秒")
                
                if hasattr(response, 'data') and isinstance(response.data, list):
                    embedding = response.data[0].embedding
                    if len(embedding) != self.dimensions:
                        raise RuntimeError(f"API返回的向量维度 ({len(embedding)}) 与预期维度 ({self.dimensions}) 不匹配")
                    return embedding
                else:
                    raise RuntimeError(f"无效的API响应格式: {response}")
            except Exception as e:
                logger.warning(f"第 {attempt + 1} 次尝试失败: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"处理智谱AI API响应时出错: {str(e)}")

    def _get_embedding(self, text: str) -> List[float]:
        """获取单个文本的嵌入向量。
        
        Args:
            text: 要生成嵌入的文本
            
        Returns:
            嵌入向量
        """
        logger.info(f"处理文本（长度：{len(text)}字符）")
        
        # 如果文本超过分块大小，则分块处理并取平均值
        if len(text) > self.chunk_size:
            chunks = self._split_text(text)
            embeddings = []
            
            for i, chunk in enumerate(tqdm(chunks, desc="处理文本块")):
                logger.info(f"处理第 {i+1}/{len(chunks)} 个文本块（长度：{len(chunk)}字符）")
                try:
                    embedding = self._get_embedding_with_retry(chunk)
                    embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"处理文本块 {i+1} 时出错: {str(e)}")
                    raise
            
            # 计算所有块的平均嵌入向量
            if embeddings:
                logger.info("计算平均嵌入向量...")
                avg_embedding = [
                    sum(values) / len(embeddings)
                    for values in zip(*embeddings)
                ]
                if len(avg_embedding) != self.dimensions:
                    raise RuntimeError(f"平均向量维度 ({len(avg_embedding)}) 与预期维度 ({self.dimensions}) 不匹配")
                return avg_embedding
            else:
                raise RuntimeError("无法生成嵌入向量")
        else:
            # 直接处理小文本
            return self._get_embedding_with_retry(text)

    def _get_embeddings_batch(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """批量获取文本嵌入向量。
        
        Args:
            texts: 要生成嵌入的文本列表
            max_retries: 最大重试次数
            
        Returns:
            嵌入向量列表
        """
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                    dimensions=self.dimensions
                )
                elapsed_time = time.time() - start_time
                logger.debug(f"批量API调用耗时: {elapsed_time:.2f}秒")
                
                if hasattr(response, 'data') and isinstance(response.data, list):
                    embeddings = [item.embedding for item in response.data]
                    # 验证所有向量维度
                    for i, emb in enumerate(embeddings):
                        if len(emb) != self.dimensions:
                            raise RuntimeError(f"第{i+1}个向量维度 ({len(emb)}) 与预期维度 ({self.dimensions}) 不匹配")
                    return embeddings
                else:
                    raise RuntimeError(f"无效的API响应格式: {response}")
            except Exception as e:
                logger.warning(f"第 {attempt + 1} 次批量处理尝试失败: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"批量处理智谱AI API响应时出错: {str(e)}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """生成文档列表的嵌入向量。
        
        Args:
            texts: 要生成嵌入的文本列表
            
        Returns:
            嵌入向量列表
        """
        logger.info(f"开始批量处理 {len(texts)} 个文档")
        results = []
        
        # 批量处理文档
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            logger.info(f"处理批次 {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1} ({len(batch_texts)} 个文档)")
            
            # 处理每个文档，如果超过chunk_size则分块
            processed_texts = []
            for text in batch_texts:
                if len(text) > self.chunk_size:
                    chunks = self._split_text(text)
                    # 对于长文本，我们取所有块的平均值
                    chunk_embeddings = self._get_embeddings_batch(chunks)
                    avg_embedding = [
                        sum(values) / len(chunk_embeddings)
                        for values in zip(*chunk_embeddings)
                    ]
                    results.append(avg_embedding)
                else:
                    processed_texts.append(text)
            
            # 批量处理标准长度的文档
            if processed_texts:
                batch_embeddings = self._get_embeddings_batch(processed_texts)
                results.extend(batch_embeddings)
        
        logger.info("文档处理完成")
        return results

    def embed_query(self, text: str) -> List[float]:
        """生成查询文本的嵌入向量。
        
        Args:
            text: 要生成嵌入的查询文本
            
        Returns:
            嵌入向量
        """
        logger.info("开始处理查询文本")
        if len(text) > self.chunk_size:
            chunks = self._split_text(text)
            chunk_embeddings = self._get_embeddings_batch(chunks)
            result = [
                sum(values) / len(chunk_embeddings)
                for values in zip(*chunk_embeddings)
            ]
        else:
            result = self._get_embeddings_batch([text])[0]
        logger.info("查询文本处理完成")
        return result 