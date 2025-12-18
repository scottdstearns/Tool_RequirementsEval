import os
from dataclasses import dataclass

@dataclass
class Settings:
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "http://litellm:4000/v1")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "sk-1234")
    chat_model: str     = os.getenv("CHAT_MODEL", "azure-gpt-4o")

    def as_dict(self):
        return {
            "openai_base_url": self.openai_base_url,
            "chat_model": self.chat_model,
        }
