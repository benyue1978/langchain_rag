"""PDF 处理器模块

此模块提供了用于处理 PDF 文件的功能，包括文本提取、清理和结构化。
"""

import re
from typing import List, Dict, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

@dataclass
class ProcessedPage:
    """表示处理后的 PDF 页面

    Attributes:
        content: 页面的文本内容
        page_number: PDF 中的页码
        contains_chapter: 是否包含章节标题
        num_paragraphs: 段落数量
    """
    content: str
    page_number: int
    contains_chapter: bool = False
    num_paragraphs: int = 0

class PDFProcessorError(Exception):
    """PDF 处理器相关的异常基类"""
    pass

class PDFLoadError(PDFProcessorError):
    """PDF 加载失败时抛出的异常"""
    pass

class PDFProcessingError(PDFProcessorError):
    """PDF 处理过程中的异常"""
    pass

class PDFProcessor:
    """PDF 文档处理器
    
    此类提供了加载和处理 PDF 文件的功能，包括文本提取、清理和结构化。
    
    Attributes:
        chapter_pattern: 用于识别章节标题的正则表达式模式
        noise_pattern: 用于识别和删除噪声文本的正则表达式模式
        char_replacements: 特殊字符替换映射
    """
    
    def __init__(self) -> None:
        """初始化 PDF 处理器"""
        # 章节模式匹配（包括中英文）
        self.chapter_pattern = re.compile(
            r'^(?:\s*)((?:Chapter|第)[.\s]*(?:\d+|[一二三四五六七八九十]+)[.\s]*(?:章)?)',
            re.IGNORECASE
        )
        
        # 噪声模式匹配
        self.noise_pattern = re.compile(
            r'^\s*$|页码|Page \d+|^\d+$|^-\d+-$|^第\s*\d+\s*页$',
            re.MULTILINE
        )
        
        # 特殊字符替换映射
        self.char_replacements: Dict[str, str] = {
            '\u3000': ' ',  # 全角空格
            '\xa0': ' ',    # 不间断空格
            '\r': '\n',     # 回车符
            '\t': ' ',      # 制表符
        }

    def load_pdf(self, pdf_path: str | Path) -> List[ProcessedPage]:
        """加载并处理 PDF 文件
        
        Args:
            pdf_path: PDF 文件路径
            
        Returns:
            包含处理后页面的列表
            
        Raises:
            PDFLoadError: 当 PDF 文件不存在或无法打开时
            PDFProcessingError: 当处理 PDF 内容时发生错误
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise PDFLoadError(f"PDF 文件不存在: {pdf_path}")
            
            doc = fitz.open(pdf_path)
            processed_pages: List[ProcessedPage] = []
            
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    content = self._process_page_content(page.get_text())
                    
                    # 检查是否包含章节标题
                    contains_chapter = bool(self.chapter_pattern.search(content))
                    
                    # 计算段落数
                    num_paragraphs = len([p for p in content.split('\n\n') if p.strip()])
                    
                    processed_pages.append(ProcessedPage(
                        content=content,
                        page_number=page_num + 1,
                        contains_chapter=contains_chapter,
                        num_paragraphs=num_paragraphs
                    ))
                    
                except Exception as e:
                    logger.error(f"处理第 {page_num + 1} 页时发生错误: {str(e)}")
                    raise PDFProcessingError(f"处理第 {page_num + 1} 页时发生错误: {str(e)}")
            
            return processed_pages
            
        except fitz.FileDataError as e:
            raise PDFLoadError(f"无法打开 PDF 文件: {str(e)}")
        except Exception as e:
            raise PDFProcessingError(f"处理 PDF 时发生错误: {str(e)}")
        finally:
            if 'doc' in locals():
                doc.close()

    def _process_page_content(self, content: str) -> str:
        """处理页面内容
        
        Args:
            content: 原始页面内容
            
        Returns:
            处理后的页面内容
        """
        # 清理特殊字符
        content = self._clean_special_chars(content)
        
        # 删除噪声
        content = self.noise_pattern.sub('', content)
        
        # 规范化空白字符
        lines = [line.strip() for line in content.split('\n')]
        content = '\n'.join(line for line in lines if line)
        
        return content

    def _clean_special_chars(self, text: str) -> str:
        """清理特殊字符
        
        Args:
            text: 要清理的文本
            
        Returns:
            清理后的文本
        """
        for old, new in self.char_replacements.items():
            text = text.replace(old, new)
        return text 