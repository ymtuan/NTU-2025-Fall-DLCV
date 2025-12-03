"""
LLM Client 抽象層
支援 Gemini (Vertex AI)、本地開源模型 (vLLM OpenAI API) 和 OpenAI API
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import os

class LLMClient(ABC):
    """LLM 客戶端抽象基類"""
    
    def __init__(self):
        self.messages = []  # 對話歷史
    
    @abstractmethod
    def send_message(self, message: str) -> str:
        """發送訊息並返回回應"""
        pass
    
    def reset_conversation(self):
        """重置對話歷史"""
        self.messages = []

class GeminiClient(LLMClient):
    """Gemini (Vertex AI) 客戶端"""
    
    def __init__(self, project_id: str, location: str = 'global', think_mode: bool = False):
        super().__init__()
        from google import genai
        
        self.client = genai.Client(vertexai=True, project=project_id, location=location)
        
        if think_mode:
            self.chat = self.client.chats.create(
                model="gemini-2.5-flash",
                config=genai.types.GenerateContentConfig(
                    thinking_config=genai.types.ThinkingConfig(thinking_budget=128), 
                    temperature=0.2
                )
            )
        else:
            self.chat = self.client.chats.create(
                model="gemini-2.5-flash",
                config=genai.types.GenerateContentConfig(temperature=0.2)
            )
    
    def send_message(self, message: str) -> str:
        """發送訊息到 Gemini"""
        self.messages.append({"role": "user", "content": message})
        response = self.chat.send_message(message)
        response_text = response.text.strip()
        self.messages.append({"role": "assistant", "content": response_text})
        return response_text

class OpenAIAPIClient(LLMClient):
    """OpenAI API 兼容客戶端（支援 vLLM）"""
    
    def __init__(self, api_base: str = None, api_key: str = "dummy", model: str = None, 
                 temperature: float = 0.2, max_tokens: int = 2048):
        """
        初始化 OpenAI API 客戶端
        
        Args:
            api_base: API 基礎 URL (例如: http://localhost:8040/v1)
            api_key: API 密鑰 (vLLM 可以使用任意值)
            model: 模型名稱
            temperature: 溫度參數
            max_tokens: 最大 token 數
        """
        super().__init__()
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=api_base
            )
        except ImportError:
            raise ImportError("請安裝 openai 套件: pip install openai")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def send_message(self, message: str) -> str:
        """發送訊息到 OpenAI API"""
        self.messages.append({"role": "user", "content": message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            response_text = response.choices[0].message.content.strip()
            self.messages.append({"role": "assistant", "content": response_text})
            return response_text
        except Exception as e:
            raise RuntimeError(f"OpenAI API 錯誤: {e}")

class LocalLLMClient(LLMClient):
    """本地開源模型客戶端（直接載入模型，不使用 API）"""
    
    def __init__(self, model_path: str = None, model_name: str = "llama", **kwargs):
        """
        初始化本地模型（直接載入）
        
        Args:
            model_path: 模型路徑
            model_name: 模型名稱 (llama, mistral, qwen, etc.)
            **kwargs: 其他模型參數
        """
        super().__init__()
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.kwargs = kwargs
        
        # TODO: 根據 model_name 載入對應的模型
        # 例如：使用 transformers 庫載入 Llama, Mistral, Qwen 等
        self._load_model()
    
    def _load_model(self):
        """載入本地模型"""
        # TODO: 實現模型載入邏輯
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        raise NotImplementedError("Direct local model loading not implemented yet. Use OpenAIAPIClient with vLLM instead.")
    
    def send_message(self, message: str) -> str:
        """發送訊息到本地模型"""
        # TODO: 實現本地模型推理邏輯
        raise NotImplementedError("Direct local model inference not implemented yet. Use OpenAIAPIClient with vLLM instead.")

# 工廠函數
def create_llm_client(client_type: str = "gemini", **kwargs) -> LLMClient:
    """
    創建 LLM 客戶端
    
    Args:
        client_type: "gemini", "openai" (vLLM), 或 "local"
        **kwargs: 客戶端特定參數
            - Gemini: project_id, location, think_mode
            - OpenAI/vLLM: api_base, api_key, model, temperature, max_tokens
            - Local: model_path, model_name
    
    Returns:
        LLMClient 實例
    """
    if client_type == "gemini":
        return GeminiClient(**kwargs)
    elif client_type == "openai" or client_type == "vllm":
        return OpenAIAPIClient(**kwargs)
    elif client_type == "local":
        return LocalLLMClient(**kwargs)
    else:
        raise ValueError(f"Unknown client type: {client_type}. Use 'gemini', 'openai'/'vllm', or 'local'")

