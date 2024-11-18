from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
# import instructor

from typing import List, Dict, Optional, Tuple, Union, AsyncGenerator
from pydantic import BaseModel

from .configs import LLMConfig
from .types import ChatMessage, DecomposedQuery
from .logging import setup_logger

logger = setup_logger(__name__)

class LLMClient:
    """Enhanced LLM client with fallback support."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.together = AsyncOpenAI(
                api_key=config.together_api_key,
                base_url=config.together_base_url
            )
        self.anthropic = AsyncAnthropic(api_key=config.anthropic_api_key)

        # TODO instructor doesnt work as expected, fails to work with ChatMessage, fails to understand ReponseModels 
        # self.together = instructor.from_openai(
        #     AsyncOpenAI(
        #         api_key=config.together_api_key,
        #         base_url=config.together_base_url
        #     )
        # )
        # self.anthropic = instructor.from_anthropic(
        #     AsyncAnthropic(api_key=config.anthropic_api_key)
        # )

    async def _format_messages_for_anthropic(
        self,
        messages: List[ChatMessage]
    ) -> Tuple[Optional[str], List[Dict[str, str]]]:
        """Format messages for Anthropic API."""
        # Extract system message
        system_msg = []
        chat_messages = []

        for m in messages:
            if m.role == "system":
                if system_msg:
                    raise ValueError("Multiple system messages found")
                system_msg.append({"role": m.role, "content": m.content})
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        return system_msg, chat_messages
    
    async def generate(
        self,
        messages: List[ChatMessage],
        stream: bool = False,
        use_primary: bool = True, # can be changed for testing 
    ) -> Union[str, AsyncGenerator[str, None], BaseModel]:
        """Generate with fallback support."""
        if use_primary:
            try:
                return await self._generate_primary(messages, stream)
            except Exception as e:
                logger.warning(f"Primary model failed: {e}, falling back to Claude")
                return await self._generate_fallback(messages, stream)
        else:
            return await self._generate_fallback(messages, stream)

    async def stream_generator(self, response) -> AsyncGenerator[str, None]:
        """Generate streaming response chunks."""
        try:
            async for chunk in response:
                if hasattr(chunk.choices[0], 'delta'):
                    # Together/OpenAI format
                    if chunk.choices[0].delta.content:  # Only yield non-empty content
                        yield chunk.choices[0].delta.content
                elif hasattr(chunk.choices[0], 'text'):
                    # Anthropic format
                    if chunk.choices[0].text:  # Only yield non-empty content
                        yield chunk.choices[0].text
        except Exception as e:
            logger.error(f"Error in stream_generator: {e}")
            raise

    async def _generate_primary(
        self,
        messages: List[ChatMessage],
        stream: bool
    ) -> Union[str, AsyncGenerator[str, None], BaseModel]:
        """Generate using Together API."""
        response = await self.together.chat.completions.create(
            model=self.config.primary_model.name,
            messages=[msg.model_dump() for msg in messages], # TODO to transform ChatMessage to dict
            temperature=self.config.primary_model.temperature,
            max_tokens=self.config.primary_model.max_tokens,
            stream=stream
        )
        
        if stream:
            return self.stream_generator(response)
        
        return response.choices[0].message.content 

    async def _generate_fallback(
        self,
        messages: List[ChatMessage],
        stream: bool
    ) -> Union[str, AsyncGenerator[str, None], BaseModel]:
        """Generate using Anthropic API."""
        system_msg, chat_messages = await self._format_messages_for_anthropic(
            messages
        )
        
        response = await self.anthropic.messages.create(
            model=self.config.fallback_model.name,
            system=system_msg,
            messages=chat_messages,
            temperature=self.config.fallback_model.temperature,
            max_tokens=self.config.fallback_model.max_tokens,
            stream=stream
        )
        
        if stream:
            return self.stream_generator(response)
        
        return response.content[0].text 

class QueryDecomposition(LLMClient):
    """Decomposes complex queries into simpler sub-queries using LLM."""

    def __init__(
        self,
        config: LLMConfig,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1
    ):
        """Initialize query decomposition with LLM client."""
        super().__init__(config)
        self.temperature = temperature
        # TODO template with a proper prompt 
        self.system_prompt = system_prompt or """You are an AI assistant specialized in decomposing complex queries into simpler sub-queries.
Your task is to:
1. Analyze the input query
2. Break it down into 2-3 focused sub-queries that together help answer the original question
3. Each sub-query should be self-contained and searchable
4. Provide brief reasoning for each sub-query
"""
        # TODO update and fix  
        # import instructor
        # from openai import OpenAI
        # from anthropic import Anthropic

        # # Initialize instructor-enabled clients
        # self.together = instructor.from_openai(
        #     OpenAI(
        #         api_key=config.together_api_key,
        #         base_url=config.together_base_url
        #     )
        # )
        # self.anthropic = instructor.from_anthropic(
        #     Anthropic(api_key=config.anthropic_api_key)
        # )

    def _create_prompt(self, query: str) -> str:
        """Create decomposition prompt."""
        return f"""Original query: {query}

Break down this search query into multiple focused sub-queries.
Think step by step about different aspects of the query that need to be searched.
Each sub-query should be self-contained and searchable."""

    async def generate(
        self,
        messages: List[ChatMessage],
        response_model: Optional[BaseModel] = None,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None], BaseModel]:
        """      
        Args:
            messages: List of chat messages
            response_model: Optional response model for structured output
            stream: Whether to stream the response
            
        Returns:
            Generated response based on response model type
        """
        try:
            if response_model:
                try:
                    # Try primary model first
                    response = await self.together.chat.completions.create(
                        model=self.config.primary_model.name,
                        messages=messages,
                        response_model=response_model
                    )
                    return response
                except Exception as e:
                    logger.warning(f"Primary model failed: {e}, falling back to Claude")
                    # Fall back to Anthropic
                    system_msg, chat_messages = await self._format_messages_for_anthropic(messages)
                    response = await self.anthropic.messages.create(
                        model=self.config.fallback_model.name,
                        system=system_msg,
                        messages=chat_messages,
                        response_model=response_model
                    )
                    return response
            else:
                # Use parent's generate method for non-response-model cases
                return await super().generate(messages, response_model, stream)
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if response_model == DecomposedQuery:
                # Special handling for DecomposedQuery failures
                query = messages[-1].content if messages else ""
                return DecomposedQuery(
                    original_query=query,
                    sub_queries=[query],
                    reasoning=f"Decomposition failed: {str(e)}"
                )
            raise  # Re-raise other exceptions
