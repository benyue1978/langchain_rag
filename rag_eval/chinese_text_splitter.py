"""中文文本分割器模块

此模块提供了专门用于中文文本分割的功能。
"""

from typing import List
from langchain.text_splitter import CharacterTextSplitter
import jieba

class ChineseTextSplitter(CharacterTextSplitter):
    """中文文本分割器
    
    此类继承自 CharacterTextSplitter，专门用于处理中文文本。
    它使用 jieba 分词来确保分割点不会破坏中文词语的完整性。
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        **kwargs
    ):
        """初始化中文文本分割器
        
        Args:
            chunk_size: 每个文本块的目标大小（字符数）
            chunk_overlap: 相邻文本块的重叠大小（字符数）
        """
        super().__init__(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            **kwargs
        )
    
    def split_text(self, text: str) -> List[str]:
        """分割文本
        
        使用 jieba 分词确保在词语边界处分割文本。
        
        Args:
            text: 要分割的文本
            
        Returns:
            分割后的文本块列表
        """
        # 使用 jieba 分词
        words = list(jieba.cut(text))
        
        # 重新组合文本，在词语之间添加空格
        text_with_word_boundaries = " ".join(words)
        
        # 使用父类的方法进行分割
        chunks = super().split_text(text_with_word_boundaries)
        
        # 移除分词时添加的空格
        return [chunk.replace(" ", "") for chunk in chunks] 