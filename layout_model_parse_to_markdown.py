# coding: utf-8
# Reference:
# https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/documentintelligence/azure-ai-documentintelligence/samples/sample_analyze_documents_output_in_markdown.py
# https://techcommunity.microsoft.com/blog/azure-ai-services-blog/build-intelligent-rag-for-multimodality-and-complex-document-structure/4118184
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
FILE: sample_analyze_documents_output_in_markdown.py

DESCRIPTION:
    This sample demonstrates how to analyze an document in markdown output format.

USAGE:
    python sample_analyze_documents_output_in_markdown.py

    Set the environment variables with your own values before running the sample:
    1) DOCUMENTINTELLIGENCE_ENDPOINT - the endpoint to your Document Intelligence resource.
    2) DOCUMENTINTELLIGENCE_API_KEY - your Document Intelligence API key.
"""
#%%
# Standard library imports
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Third-party imports
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from dotenv import load_dotenv
from PIL import Image
import fitz
import base64
from openai import AzureOpenAI
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat, AnalyzeResult

#%% Configure logging
logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Azure SDKのログレベルを警告以上に設定
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

#%%
class DocumentAnalysisError(Exception):
    """ドキュメント解析中のエラーを処理する基本例外クラス。"""
    pass

#%%
@dataclass
class FigureMetadata:
    """図のメタデータを管理するデータクラス。

    Attributes:
        page_number: 図が存在するページ番号
        paragraphs: 関連する段落のリスト
        caption: 図のキャプション
        saved_image: 保存された画像のパス
        coordinates: 図の座標情報
    """
    page_number: Optional[int] = None
    paragraphs: List[str] = None
    caption: Optional[str] = None
    saved_image: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None
    analysis: Optional[str] = None  # 追加: GPT-4による解析結果

    def __post_init__(self):
        """初期化後の処理"""
        if self.paragraphs is None:
            self.paragraphs = []
        if self.coordinates is None:
            self.coordinates = {}

#%%
def local_image_to_data_url(image_path: str) -> str:
    """画像ファイルをdata URLに変換する."""
    with open(image_path, "rb") as image_file:
        return f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"

class DocumentAnalyzer:
    """PDFドキュメントの解析と図の抽出を行うクラス。

    Attributes:
        client: Document Intelligence クライアント
        output_dir: 出力ディレクトリのパス
    """

    def __init__(self, pdf_path: str) -> None:
        """DocumentAnalyzerの初期化。

        Args:
            pdf_path: 解析対象のPDFファイルパス

        Raises:
            DocumentAnalysisError: 環境変数が未設定の場合
        """
        load_dotenv()
        
        endpoint = os.getenv("AZURE_DI_ENDPOINT")
        key = os.getenv("AZURE_DI_KEY")
        
        if not endpoint or not key:
            raise DocumentAnalysisError("Required environment variables are not set")
        
        self.client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )
        
        # 出力ディレクトリをカレントディレクトリ配下に変更
        current_dir = Path.cwd()
        self.output_dir = current_dir / "extracted_figures"
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("インスタンスを初期化しています...")
        logger.info(f"出力ディレクトリを作成しました: {self.output_dir}")
        self._markdown_content = None  # Markdown content を保持する変数を追加
        self._figure_metadata = []  # 図のメタデータを保持するリストを追加

        # Azure OpenAI関連の設定を更新
        self.openai_api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.gpt_deployment = os.getenv("GPT_DEPLOYMENT_NAME", "gpt-4o")
        self.openai_api_version = "2024-02-15-preview"
        
        if not all([self.openai_api_base, self.openai_api_key]):
            raise DocumentAnalysisError("Required Azure OpenAI environment variables are not set")
        
        self.openai_client = AzureOpenAI(
            api_key=self.openai_api_key,
            api_version=self.openai_api_version,
            base_url=f"{self.openai_api_base}/openai/deployments/{self.gpt_deployment}"
        )

    def _extract_bbox(self, polygon: List[float]) -> Dict[str, float]:
        """ポリゴン座標からバウンディングボックスを計算。

        Args:
            polygon: ポリゴンの座標リスト [x1, y1, x2, y2, ...]

        Returns:
            バウンディングボックスの座標辞書
        """
        points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
        return {
            'left': min(p[0] for p in points),
            'top': min(p[1] for p in points),
            'right': max(p[0] for p in points),
            'bottom': max(p[1] for p in points)
        }

    def _crop_image(self, pdf_path: str, page_num: int, bbox: Dict[str, float]) -> Image.Image:
        """PDFから画像領域を切り出す。

        Args:
            pdf_path: PDFファイルのパス
            page_num: ページ番号
            bbox: バウンディングボックスの座標

        Returns:
            切り出された画像

        Raises:
            DocumentAnalysisError: 画像の切り出しに失敗した場合
        """
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num)
            
            coords = (
                bbox['left'], bbox['top'],
                bbox['right'], bbox['bottom']
            )
            rect = fitz.Rect([x * 72 for x in coords])
            
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), clip=rect)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            doc.close()
            return img
            
        except Exception as e:
            raise DocumentAnalysisError(f"Failed to crop image: {e}")

    def _analyze_image_with_gpt(self, image_path: str, caption: Optional[str] = None) -> str:
        """画像をGPT-4o APIを使用して解析する.
        
        Args:
            image_path: 画像ファイルのパス
            caption: 画像のキャプション（オプション）

        Returns:
            解析結果のテキスト
        """
        try:
            data_url = local_image_to_data_url(image_path)
            
            # プロンプトを構築
            system_message = """あなたは研究論文の図表を解析する専門家です。
            Transformerアーキテクチャに関する深い知識を持ち、図表の技術的な詳細を
            読者に分かりやすく説明することができます。"""
            
            user_message = """この図について以下の点を詳しく説明してください：
            1. 図が示している内容の要約
            2. 主要なコンポーネントとその関係性
            3. 図から読み取れる重要なポイント
            4. Transformerアーキテクチャにおける位置づけと重要性"""
            
            if caption:
                user_message = f"この図（キャプション: {caption}）について説明してください:\n{user_message}"

            response = self.openai_client.chat.completions.create(
                model=self.gpt_deployment,  # 環境変数から取得したデプロイメント名を使用
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_message
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"画像解析に失敗: {e}")
            return "画像の解析に失敗しました。"

    def analyze_document(self, pdf_path: str) -> List[FigureMetadata]:
        """PDFドキュメントを解析して図を抽出。"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        try:
            logger.info(f"PDFファイル '{pdf_path}' の解析を開始します")
            
            # ファイルの読み込みとドキュメント解析の実行
            with open(pdf_path, "rb") as f:
                file_content = f.read()
            
            poller = self.client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=file_content,
                content_type="application/pdf",
                output_content_format=DocumentContentFormat.MARKDOWN,
            )
            result: AnalyzeResult = poller.result()
            
            # Markdownコンテンツを保持
            self._markdown_content = result.content
            self._figure_metadata = []  # 図のメタデータを保持するリストを初期化
            
            if not hasattr(result, 'figures'):
                logger.info("ドキュメント内に図は検出されませんでした")
                return []

            total_figures = len(result.figures)
            logger.info(f"合計 {total_figures} 個の図を検出しました")
            
            figures_metadata = []
            for idx, figure in enumerate(result.figures, 1):
                logger.info(f"=== 図 {idx}/{total_figures} の処理中 ===")
                metadata = FigureMetadata()
                
                if hasattr(figure, 'caption'):
                    caption = getattr(figure.caption, 'content', None)
                    if caption:
                        logger.info(f"キャプション: {caption[:100]}...")
                        metadata.caption = caption

                if hasattr(figure, 'elements'):
                    para_count = len([e for e in figure.elements if 'paragraphs' in e])
                    logger.info(f"関連する段落数: {para_count}")
                    metadata.paragraphs = [
                        elem for elem in figure.elements 
                        if 'paragraphs' in elem
                    ]

                if hasattr(figure, 'bounding_regions'):
                    for region in figure.bounding_regions:
                        try:
                            page_num = region.page_number
                            logger.info(f"ページ {page_num} から図を抽出しています...")
                            
                            metadata.page_number = page_num
                            metadata.coordinates = self._extract_bbox(region.polygon)
                            
                            img = self._crop_image(
                                pdf_path,
                                page_num - 1,
                                metadata.coordinates
                            )
                            
                            img_filename = f"figure_{idx}_page{page_num}.png"
                            if metadata.caption:
                                safe_caption = "".join(
                                    c for c in metadata.caption[:30] 
                                    if c.isalnum() or c in (' ', '-', '_')
                                )
                                img_filename = f"figure_{idx}_page{page_num}_{safe_caption}.png"
                            
                            img_path = self.output_dir / img_filename
                            img.save(img_path)
                            metadata.saved_image = str(img_path)
                            logger.info(f"画像を保存しました: {img_filename}")

                            # 画像の解析を実行
                            logger.info(f"GPT-4oで画像を解析しています: {img_filename}")
                            analysis_result = self._analyze_image_with_gpt(str(img_path), metadata.caption)
                            metadata.analysis = analysis_result
                            logger.info(f"画像の解析が完了しました")
                            
                        except Exception as e:
                            logger.error(f"図の抽出に失敗: {e}")
                            continue

                figures_metadata.append(metadata)

            logger.info(f"図の抽出が完了しました。合計: {len(figures_metadata)} 個")
            self._figure_metadata = figures_metadata  # メタデータを保存
            return figures_metadata
            
        except AzureError as e:
            logger.error(f"Azure service error: {e}")
            raise DocumentAnalysisError(f"Azure service error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during analysis: {e}")
            raise DocumentAnalysisError(f"Analysis failed: {str(e)}")

    def get_markdown_content(self) -> Optional[str]:
        """解析結果のMarkdownコンテンツを取得し、図の解析結果を挿入。"""
        if not self._markdown_content:
            return None

        modified_content = []
        current_content = self._markdown_content.split('\n')
        i = 0
        figure_count = 0  # 図のカウンターを追加

        while i < len(current_content):
            line = current_content[i]
            if line.startswith('<figure>'):
                figure_count += 1  # 図のカウントを増やす
                figure_content = ['<figure>']
                i += 1
                
                has_analysis = False  # 解析結果が挿入されたかどうかを追跡
                caption = None
                
                # 図の内容を収集
                while i < len(current_content) and not current_content[i].startswith('</figure>'):
                    current_line = current_content[i]
                    figure_content.append(current_line)

                    if current_line.startswith('<figcaption>'):
                        caption = current_line.replace('<figcaption>', '').replace('</figcaption>', '')
                    
                    # 図の内容の終わりを検出
                    if i + 1 >= len(current_content) or current_content[i + 1].startswith('</figure>'):
                        # 対応するメタデータを検索
                        for metadata in self._figure_metadata:
                            if metadata.caption == caption and hasattr(metadata, 'analysis'):
                                # 解析結果を追加
                                figure_content.extend([
                                    '',  # 空行
                                    '### GPT-4oによる解析:',
                                    metadata.analysis,
                                    ''   # 空行
                                ])
                                has_analysis = True
                                break
                        
                        # キャプションが見つからない場合は図番号で対応を試みる
                        if not has_analysis and figure_count <= len(self._figure_metadata):
                            metadata = self._figure_metadata[figure_count - 1]
                            if hasattr(metadata, 'analysis'):
                                figure_content.extend([
                                    '',
                                    '### GPT-4oによる解析:',
                                    metadata.analysis,
                                    ''
                                ])
                                has_analysis = True
                    
                    i += 1

                if i < len(current_content):  # </figure>タグを追加
                    figure_content.append('</figure>')
                    i += 1
                
                modified_content.extend(figure_content)
                
                # デバッグ情報を記録
                if not has_analysis:
                    logger.warning(f"Figure {figure_count} (caption: {caption}) の解析結果が挿入されませんでした")
            else:
                modified_content.append(line)
                i += 1
        
        return '\n'.join(modified_content)

#%%
def main() -> None:
    """メイン実行関数。"""
    try:
        pdf_path = "./Attention_is_All_You_Need.pdf"
        
        analyzer = DocumentAnalyzer(pdf_path)
        figures = analyzer.analyze_document(pdf_path)
        
        # Markdownファイルをカレントディレクトリに保存
        markdown_content = analyzer.get_markdown_content()
        if markdown_content:
            markdown_path = Path.cwd() / "analysis_output.md"
            with open(markdown_path, "w", encoding="utf-8") as out_file:
                out_file.write(markdown_content)
            logger.info(f"Markdown解析結果を保存しました: {markdown_path}")
        
        logger.info(f"解析完了。{len(figures)}個の図を検出しました")
        
    except (DocumentAnalysisError, FileNotFoundError) as e:
        logger.error(f"解析失敗: {e}")
        raise
    except Exception as e:
        logger.critical(f"予期せぬエラー: {e}")
        raise

#%%
if __name__ == "__main__":
    main()
#%%