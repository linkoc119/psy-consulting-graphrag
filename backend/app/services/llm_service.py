"""
LLM Service for GraphRAG Psychology Chatbot
Supports both Ollama and Anthropic Claude API
"""
import logging
import aiohttp
import asyncio
from typing import AsyncGenerator, Optional, List, Dict, Any
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseLLM:
    """Base LLM interface"""

    async def generate(
        self,
        prompt: str,
        stream: bool = True,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        raise NotImplementedError

    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        full_response = ""
        async for chunk in self.generate(prompt, stream=True, system_prompt=system_prompt):
            full_response += chunk
        return full_response

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        raise NotImplementedError

    async def check_connection(self) -> bool:
        raise NotImplementedError

    async def close(self):
        pass


class OllamaLLM(BaseLLM):
    """Ollama LLM wrapper (legacy)"""

    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model: str = settings.OLLAMA_MODEL,
        temperature: float = settings.LLM_TEMPERATURE,
        max_tokens: int = settings.LLM_MAX_TOKENS
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def generate(
        self,
        prompt: str,
        stream: bool = True,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        session = await self._ensure_session()

        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": stream,
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            **kwargs
        }

        url = f"{self.base_url}/api/generate"

        try:
            logger.debug(f"Generating with Ollama {self.model}, stream={stream}")
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama error {response.status}: {error_text}")
                    raise RuntimeError(f"Ollama API error: {error_text}")

                if stream:
                    async for line in response.content:
                        if line:
                            try:
                                data = line.decode('utf-8').strip()
                                if data:
                                    import json
                                    chunk_data = json.loads(data)
                                    if 'response' in chunk_data:
                                        yield chunk_data['response']
                            except Exception as e:
                                logger.debug(f"Error parsing stream chunk: {e}")
                                continue
                else:
                    response_text = await response.text()
                    import json
                    data = json.loads(response_text)
                    yield data.get('response', '')

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield f"❌ Lỗi: {str(e)}"

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        prompt_parts = []

        if system_prompt:
            prompt_parts.append(f"SYSTEM: {system_prompt}")

        for msg in messages:
            role = msg.get('role', 'user').upper()
            content = msg.get('content', '')
            prompt_parts.append(f"{role}: {content}")

        prompt_parts.append("ASSISTANT:")
        full_prompt = "\n\n".join(prompt_parts)

        return await self.generate_text(full_prompt)

    async def check_connection(self) -> bool:
        try:
            session = await self._ensure_session()
            url = f"{self.base_url}/api/tags"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m['name'] for m in data.get('models', [])]
                    if self.model in models or any(self.model in m for m in models):
                        logger.info(f"✅ Ollama is running with model {self.model}")
                        return True
                    else:
                        logger.warning(f"Model {self.model} not found. Available: {models}")
                        return False
                else:
                    logger.error(f"Ollama connection failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            return False

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()


class ClaudeLLM(BaseLLM):
    """Anthropic Claude API wrapper using Messages API"""

    def __init__(
        self,
        api_key: str = settings.ANTHROPIC_API_KEY,
        api_base: str = settings.ANTHROPIC_API_BASE,
        model: str = settings.ANTHROPIC_MODEL,
        temperature: float = settings.LLM_TEMPERATURE,
        max_tokens: int = settings.LLM_MAX_TOKENS
    ):
        self.api_key = api_key
        self.api_base = api_base.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def generate(
        self,
        prompt: str,
        stream: bool = True,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        session = await self._ensure_session()

        # Build messages array for Claude Messages API
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "anthropic-version": "2023-06-01"
        }

        url = f"{self.api_base}/messages"

        try:
            logger.debug(f"Generating with Claude {self.model}, stream={stream}")
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Claude API error {response.status}: {error_text}")
                    raise RuntimeError(f"Claude API error: {error_text}")

                if stream:
                    async for line in response.content:
                        if line:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: '):
                                data = line_str[6:]  # Remove "data: " prefix
                                if data == '[DONE]':
                                    break
                                try:
                                    import json
                                    chunk_data = json.loads(data)
                                    if 'delta' in chunk_data and 'text' in chunk_data['delta']:
                                        yield chunk_data['delta']['text']
                                except Exception as e:
                                    logger.debug(f"Error parsing stream chunk: {e}")
                                    continue
                else:
                    response_text = await response.text()
                    import json
                    data = json.loads(response_text)
                    if 'content' in data and len(data['content']) > 0:
                        yield data['content'][0].get('text', '')

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield f"❌ Lỗi: {str(e)}"

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """Chat with conversation history"""
        claude_messages = []

        if system_prompt:
            claude_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg.get('role', 'user')
            # Claude expects 'user' or 'assistant'
            if role not in ['user', 'assistant']:
                role = 'user'
            claude_messages.append({
                "role": role,
                "content": msg.get('content', '')
            })

        payload = {
            "model": self.model,
            "messages": claude_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "anthropic-version": "2023-06-01"
        }

        url = f"{self.api_base}/messages"

        try:
            session = await self._ensure_session()
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Claude API error {response.status}: {error_text}")
                    raise RuntimeError(f"Claude API error: {error_text}")

                data = await response.json()
                if 'content' in data and len(data['content']) > 0:
                    return data['content'][0].get('text', '')
                return ""

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"❌ Lỗi: {str(e)}"

    async def check_connection(self) -> bool:
        try:
            session = await self._ensure_session()
            # Claude API doesn't have a simple health check endpoint
            # We'll do a minimal request to test connection
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "anthropic-version": "2023-06-01"
            }
            url = f"{self.api_base}/messages"
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    logger.info(f"✅ Claude API is running with model {self.model}")
                    return True
                else:
                    logger.error(f"Claude API connection failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Cannot connect to Claude API: {e}")
            return False

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()


# Singleton instances
_llm_instance: Optional[BaseLLM] = None


def get_llm() -> BaseLLM:
    """Get or create LLM instance based on configured provider"""
    global _llm_instance
    if _llm_instance is None:
        provider = settings.LLM_PROVIDER
        if provider == "ollama":
            _llm_instance = OllamaLLM()
        elif provider == "claude":
            _llm_instance = ClaudeLLM()
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    return _llm_instance


# For testing
async def test_llm():
    """Test LLM connection"""
    llm = get_llm()
    connected = await llm.check_connection()
    if not connected:
        print("❌ LLM not available")
        return
    print(f"✅ LLM connected: {settings.LLM_PROVIDER}")

    prompt = "Hãy liệt kê 3 loại thuốc hạ sốt phổ biến:"
    print(f"\nPrompt: {prompt}")
    print("\nResponse:")
    async for chunk in llm.generate(prompt, stream=True):
        print(chunk, end='', flush=True)
    print("\n")


if __name__ == "__main__":
    asyncio.run(test_llm())