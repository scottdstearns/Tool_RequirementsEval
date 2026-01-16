import os
from dataclasses import dataclass

@dataclass
class Settings:
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "http://litellm:4000/v1")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "sk-1234")
    chat_model: str     = os.getenv("CHAT_MODEL", "azure-gpt-4o")

    def __post_init__(self):
        """Validate and log configuration at startup"""
        print("\n" + "="*60)
        print("🔧 Application Configuration")
        print("="*60)
        print(f"OpenAI Base URL: {self.openai_base_url}")
        print(f"Chat Model: {self.chat_model}")
        # Mask API key for security (show only last 4 chars)
        masked_key = f"{'*' * 10}{self.openai_api_key[-4:]}" if len(self.openai_api_key) > 4 else "****"
        print(f"API Key: {masked_key}")
        print("="*60 + "\n")
        
        # Basic validation
        if not self.openai_base_url:
            print("⚠️  WARNING: OPENAI_BASE_URL is not set")
        if not self.chat_model:
            print("⚠️  WARNING: CHAT_MODEL is not set")

    def as_dict(self):
        return {
            "openai_base_url": self.openai_base_url,
            "chat_model": self.chat_model,
        }
