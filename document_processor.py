"""PDF文档预处理模块

此模块提供了一系列用于优化PDF文档内容的工具函数，包括文本清理、结构提取等功能。
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
import datetime


@dataclass
class ProcessedDocument:
    """处理后的文档结构"""
    content: str
    metadata: Dict
    structure: Dict


class DocumentPreprocessor:
    """文档预处理器类"""
    
    def __init__(self):
        # 常见的页眉页脚模式
        self.header_footer_patterns = [
            r'^\d+$',  # 纯数字页码
            r'^第\d+页$',  # 中文页码
            r'^Page \d+$',  # 英文页码
            r'^\d+/\d+$',  # 分数式页码
            r'^[\-\=]{3,}$',  # 分隔线
        ]
        
        # 文档结构标识符
        self.structure_patterns = {
            'title': r'^第[一二三四五六七八九十]+章[\s\S]*$',
            'subtitle': r'^[\d\.]+[\s\S]*$',
        }
    
    def clean_text(self, text: str) -> str:
        """清理文本内容
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 修复错误的换行
        text = re.sub(r'(?<=[^。！？\.\!\?])\n', '', text)
        
        # 规范化标点符号
        text = re.sub(r'[，,]', '，', text)
        text = re.sub(r'[。.]', '。', text)
        text = re.sub(r'[！!]', '！', text)
        text = re.sub(r'[？?]', '？', text)
        
        return text
    
    def remove_headers_footers(self, text: str) -> str:
        """移除页眉页脚
        
        Args:
            text: 原始文本
            
        Returns:
            移除页眉页脚后的文本
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # 跳过匹配页眉页脚模式的行
            if any(re.match(pattern, line) for pattern in self.header_footer_patterns):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def extract_structure(self, text: str) -> Tuple[str, Dict]:
        """提取文档结构
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的文本和结构信息的元组
        """
        lines = text.split('\n')
        structure = {
            'titles': [],
            'subtitles': [],
            'paragraphs': []
        }
        
        processed_lines = []
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_paragraph:
                    structure['paragraphs'].append(' '.join(current_paragraph))
                    current_paragraph = []
                continue
                
            # 检查是否为标题
            if re.match(self.structure_patterns['title'], line):
                structure['titles'].append(line)
                if current_paragraph:
                    structure['paragraphs'].append(' '.join(current_paragraph))
                    current_paragraph = []
                processed_lines.append(line)
            # 检查是否为子标题
            elif re.match(self.structure_patterns['subtitle'], line):
                structure['subtitles'].append(line)
                if current_paragraph:
                    structure['paragraphs'].append(' '.join(current_paragraph))
                    current_paragraph = []
                processed_lines.append(line)
            else:
                current_paragraph.append(line)
                
        if current_paragraph:
            structure['paragraphs'].append(' '.join(current_paragraph))
            
        return '\n'.join(processed_lines), structure
    
    def process_document(self, document: Document) -> ProcessedDocument:
        """处理单个文档
        
        Args:
            document: 原始文档对象
            
        Returns:
            处理后的文档对象
        """
        # 获取原始文本
        text = document.page_content
        
        # 移除页眉页脚
        text = self.remove_headers_footers(text)
        
        # 提取文档结构
        text, structure = self.extract_structure(text)
        
        # 清理文本
        text = self.clean_text(text)
        
        # 更新元数据（扁平化结构）
        metadata = document.metadata.copy()
        metadata.update({
            'processed': True,
            'processing_timestamp': datetime.datetime.now().isoformat(),
            'num_titles': len(structure['titles']),
            'num_subtitles': len(structure['subtitles']),
            'num_paragraphs': len(structure['paragraphs']),
            'first_title': structure['titles'][0] if structure['titles'] else '',
            'has_structure': bool(structure['titles'] or structure['subtitles'])
        })
        
        return ProcessedDocument(
            content=text,
            metadata=metadata,
            structure=structure
        )
    
    def batch_process(self, documents: List[Document]) -> List[ProcessedDocument]:
        """批量处理文档
        
        Args:
            documents: 文档列表
            
        Returns:
            处理后的文档列表
        """
        return [self.process_document(doc) for doc in documents] 