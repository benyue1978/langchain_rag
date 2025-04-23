from typing import TYPE_CHECKING, List
import os
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

from embeddings import ZhipuAIEmbeddings

@dataclass
class MockEmbedding:
    """模拟Embedding响应对象。"""
    object: str = "embedding"
    index: int = 0
    embedding: List[float] = None

    def __post_init__(self):
        if self.embedding is None:
            # 创建2048维的模拟向量
            self.embedding = [0.1] * 2048

@dataclass
class MockEmbeddingsResponse:
    """模拟EmbeddingsResponded响应对象。"""
    object: str = "list"
    data: List[MockEmbedding] = None

    def __post_init__(self):
        if self.data is None:
            self.data = [MockEmbedding()]

@pytest.fixture
def mock_zhipuai_response() -> MockEmbeddingsResponse:
    """模拟智谱AI API的响应。"""
    return MockEmbeddingsResponse()

@pytest.fixture
def mock_client(mock_zhipuai_response: MockEmbeddingsResponse) -> MagicMock:
    """创建模拟的智谱AI客户端。"""
    mock = MagicMock()
    mock.embeddings.create.return_value = mock_zhipuai_response
    return mock

def test_init_with_api_key() -> None:
    """测试使用API密钥初始化。"""
    embeddings = ZhipuAIEmbeddings(api_key="test_key")
    assert embeddings.api_key == "test_key"
    assert embeddings.model == "embedding-3"
    assert embeddings.dimensions == 2048  # 验证默认维度
    assert embeddings.chunk_size == 2000

def test_init_with_custom_dimensions() -> None:
    """测试自定义维度。"""
    embeddings = ZhipuAIEmbeddings(api_key="test_key", dimensions=1536)
    assert embeddings.dimensions == 1536

def test_init_with_custom_chunk_size() -> None:
    """测试自定义分块大小。"""
    embeddings = ZhipuAIEmbeddings(api_key="test_key", chunk_size=1000)
    assert embeddings.chunk_size == 1000

def test_init_with_env_var(monkeypatch: "MonkeyPatch") -> None:
    """测试使用环境变量初始化。"""
    monkeypatch.setenv("ZHIPUAI_API_KEY", "env_key")
    embeddings = ZhipuAIEmbeddings()
    assert embeddings.api_key == "env_key"

def test_init_without_api_key(monkeypatch: "MonkeyPatch") -> None:
    """测试没有API密钥时的错误处理。"""
    monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="智谱AI API密钥未设置"):
        ZhipuAIEmbeddings()

@patch("embeddings.ZhipuAI")
def test_embed_documents(mock_zhipuai_cls: MagicMock, mock_client: MagicMock, mock_zhipuai_response: MockEmbeddingsResponse) -> None:
    """测试文档嵌入功能。"""
    mock_zhipuai_cls.return_value = mock_client
    
    embeddings = ZhipuAIEmbeddings(api_key="test_key")
    texts = ["测试文本1", "测试文本2"]
    result = embeddings.embed_documents(texts)
    
    assert len(result) == 2
    assert all(isinstance(embedding, list) for embedding in result)
    assert all(len(embedding) == 2048 for embedding in result)  # 验证输出维度
    assert mock_client.embeddings.create.call_count == 2

@patch("embeddings.ZhipuAI")
def test_embed_query(mock_zhipuai_cls: MagicMock, mock_client: MagicMock, mock_zhipuai_response: MockEmbeddingsResponse) -> None:
    """测试查询嵌入功能。"""
    mock_zhipuai_cls.return_value = mock_client
    
    embeddings = ZhipuAIEmbeddings(api_key="test_key")
    result = embeddings.embed_query("测试查询")
    
    assert isinstance(result, list)
    assert len(result) == 2048  # 验证输出维度
    mock_client.embeddings.create.assert_called_once()

@patch("embeddings.ZhipuAI")
def test_embed_with_dimensions(mock_zhipuai_cls: MagicMock, mock_client: MagicMock, mock_zhipuai_response: MockEmbeddingsResponse) -> None:
    """测试指定维度的嵌入功能。"""
    mock_zhipuai_cls.return_value = mock_client
    
    embeddings = ZhipuAIEmbeddings(api_key="test_key", dimensions=2048)
    embeddings.embed_query("测试文本")
    
    mock_client.embeddings.create.assert_called_with(
        model="embedding-3",
        input="测试文本",
        dimensions=2048
    )

@patch("embeddings.ZhipuAI")
def test_api_error_handling(mock_zhipuai_cls: MagicMock, mock_client: MagicMock) -> None:
    """测试API错误处理。"""
    mock_client.embeddings.create.return_value = None
    mock_zhipuai_cls.return_value = mock_client
    
    embeddings = ZhipuAIEmbeddings(api_key="test_key")
    with pytest.raises(RuntimeError, match="无效的API响应格式"):
        embeddings.embed_query("测试文本")

@patch("embeddings.ZhipuAI")
def test_dimension_mismatch_handling(mock_zhipuai_cls: MagicMock, mock_client: MagicMock) -> None:
    """测试维度不匹配的错误处理。"""
    # 创建一个维度不匹配的响应
    wrong_response = MockEmbeddingsResponse()
    wrong_response.data[0].embedding = [0.1] * 1536  # 使用1536维而不是2048维
    mock_client.embeddings.create.return_value = wrong_response
    mock_zhipuai_cls.return_value = mock_client
    
    embeddings = ZhipuAIEmbeddings(api_key="test_key")
    with pytest.raises(RuntimeError, match="API返回的向量维度.*与预期维度.*不匹配"):
        embeddings.embed_query("测试文本")

@patch("embeddings.ZhipuAI")
def test_long_text_chunking(mock_zhipuai_cls: MagicMock, mock_client: MagicMock, mock_zhipuai_response: MockEmbeddingsResponse) -> None:
    """测试长文本分块处理。"""
    mock_zhipuai_cls.return_value = mock_client
    
    # 创建一个超过分块大小的文本
    long_text = "测试" * 1000  # 2000个字符
    embeddings = ZhipuAIEmbeddings(api_key="test_key", chunk_size=500)  # 设置较小的分块大小
    result = embeddings.embed_query(long_text)
    
    # 验证结果
    assert isinstance(result, list)
    assert len(result) == 2048  # 验证输出维度
    # 验证API被调用了4次（2000字符/500字符每块 = 4块）
    assert mock_client.embeddings.create.call_count == 4 