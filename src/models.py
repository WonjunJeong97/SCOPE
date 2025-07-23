# src/models.py
"""
Model wrappers for different LLM APIs.
Provides a unified interface for OpenAI, Claude, Gemini, and LLaMA (via Groq).
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

# API clients
from openai import OpenAI
import anthropic
import google.generativeai as genai
from groq import Groq

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all language models."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or self._get_api_key()
        self.client = self._initialize_client()
        
    @abstractmethod
    def _get_api_key(self) -> str:
        """Get API key from environment variables."""
        pass
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the API client."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1, temperature: float = 1.0) -> str:
        """Generate response from the model."""
        pass
    
    def _handle_rate_limit(self, retry_after: float = 1.0):
        """Handle rate limiting with exponential backoff."""
        logger.warning(f"Rate limit hit, waiting {retry_after} seconds...")
        time.sleep(retry_after)


class OpenAIModel(BaseModel):
    """OpenAI model wrapper (ChatGPT)."""
    
    def _get_api_key(self) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return api_key
    
    def _initialize_client(self):
        return OpenAI(api_key=self.api_key)
    
    def generate(self, prompt: str, max_tokens: int = 1, temperature: float = 1.0) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer with just the letter."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                n=1
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            if "rate_limit" in str(e).lower():
                self._handle_rate_limit()
                return self.generate(prompt, max_tokens, temperature)
            else:
                logger.error(f"OpenAI API error: {e}")
                raise


class ClaudeModel(BaseModel):
    """Anthropic Claude model wrapper."""
    
    def _get_api_key(self) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return api_key
    
    def _initialize_client(self):
        return anthropic.Anthropic(api_key=self.api_key)
    
    def generate(self, prompt: str, max_tokens: int = 1, temperature: float = 1.0) -> str:
        try:
            # Map model names to Anthropic's naming convention
            model_mapping = {
                "claude-3-haiku": "claude-3-haiku-20240307",
                "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
                "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
            }
            
            model = model_mapping.get(self.model_name, self.model_name)
            
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system="You are a helpful assistant. Answer with just the letter.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
            
        except Exception as e:
            if "rate_limit" in str(e).lower():
                self._handle_rate_limit()
                return self.generate(prompt, max_tokens, temperature)
            else:
                logger.error(f"Claude API error: {e}")
                raise


class GeminiModel(BaseModel):
    """Google Gemini model wrapper."""
    
    def _get_api_key(self) -> str:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        return api_key
    
    def _initialize_client(self):
        genai.configure(api_key=self.api_key)
        
        # Map model names
        model_mapping = {
            "gemini-1.5-flash": "gemini-1.5-flash",
            "gemini-1.5-pro": "gemini-1.5-pro",
        }
        
        model_name = model_mapping.get(self.model_name, self.model_name)
        return genai.GenerativeModel(model_name)
    
    def generate(self, prompt: str, max_tokens: int = 1, temperature: float = 1.0) -> str:
        try:
            # Gemini uses different parameter names
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Add system prompt to the beginning
            full_prompt = "You are a helpful assistant. Answer with just the letter.\n\n" + prompt
            
            response = self.client.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            return response.text.strip()
            
        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                self._handle_rate_limit()
                return self.generate(prompt, max_tokens, temperature)
            else:
                logger.error(f"Gemini API error: {e}")
                raise


class GroqModel(BaseModel):
    """Groq model wrapper (for LLaMA models)."""
    
    def _get_api_key(self) -> str:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        return api_key
    
    def _initialize_client(self):
        return Groq(api_key=self.api_key)
    
    def generate(self, prompt: str, max_tokens: int = 1, temperature: float = 1.0) -> str:
        try:
            # Map model names to Groq's naming convention
            model_mapping = {
                "llama-3-8b": "llama3-8b-8192",
                "llama-3-70b": "llama3-70b-8192",
                "llama3-8b": "llama3-8b-8192",
                "llama3-70b": "llama3-70b-8192",
            }
            
            model = model_mapping.get(self.model_name, self.model_name)
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer with just the letter."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                n=1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            if "rate_limit" in str(e).lower():
                self._handle_rate_limit()
                return self.generate(prompt, max_tokens, temperature)
            else:
                logger.error(f"Groq API error: {e}")
                raise


def get_model(model_name: str, api_key: Optional[str] = None) -> BaseModel:
    """
    Factory function to get the appropriate model instance.
    
    Args:
        model_name: Name of the model
        api_key: Optional API key (will use env var if not provided)
        
    Returns:
        Model instance
    """
    # Model to provider mapping
    model_providers = {
        # OpenAI
        "gpt-3.5-turbo": OpenAIModel,
        "gpt-4": OpenAIModel,
        "gpt-4o-mini": OpenAIModel,
        "gpt-4o": OpenAIModel,
        
        # Claude
        "claude-3-haiku": ClaudeModel,
        "claude-3-5-sonnet": ClaudeModel,
        "claude-3.5-sonnet": ClaudeModel,
        
        # Gemini
        "gemini-1.5-flash": GeminiModel,
        "gemini-1.5-pro": GeminiModel,
        
        # LLaMA via Groq
        "llama-3-8b": GroqModel,
        "llama-3-70b": GroqModel,
        "llama3-8b": GroqModel,
        "llama3-70b": GroqModel,
    }
    
    if model_name not in model_providers:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_providers.keys())}")
    
    model_class = model_providers[model_name]
    return model_class(model_name, api_key)


# Example usage
if __name__ == "__main__":
    # Test with different models
    test_prompt = """You must choose one. If you had to pick, which would it be?
    xyzabc
    qwerty
    mnbvcx
    asdfgh"""
    
    # Test OpenAI
    try:
        model = get_model("gpt-3.5-turbo")
        response = model.generate(test_prompt)
        print(f"GPT-3.5-turbo response: {response}")
    except Exception as e:
        print(f"OpenAI test failed: {e}")
    
    # Test Claude
    try:
        model = get_model("claude-3-haiku")
        response = model.generate(test_prompt)
        print(f"Claude-3-haiku response: {response}")
    except Exception as e:
        print(f"Claude test failed: {e}")