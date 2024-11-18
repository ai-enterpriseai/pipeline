"""
generator.py
-----------
Enhanced generation module with advanced features.
"""
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from enum import Enum

from pydantic import BaseModel, Field

from .retriever import Retriever 
from .utils.model import LLMClient 
from .utils.types import ChatMessage  
from .utils.logging import setup_logger

logger = setup_logger(__name__)

class ConversationMode(str, Enum):
    STANDARD = "test_template"
    ACADEMIC = "academic"

class Memory(BaseModel):
    """Tracks the current state of the conversation."""
    messages: List[ChatMessage] = Field(default_factory=list)
    current_template: Optional[str] = None
    
    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the conversation state."""
        self.messages.append(message)
    
    def get_messages(self) -> List[ChatMessage]:
        """Get all messages in the conversation."""
        return self.messages.copy()

class Generator:
    """Enhanced generator with improved message handling."""
    from .prompts.manager import PromptManager

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_manager: PromptManager,
        retriever: Retriever 
    ):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.retriever = retriever
        self.memory = Memory()
    
    async def _get_system_prompt(self, template_name: str, variables: Optional[Dict[str, Any]] = None) -> str:
        """Get formatted system prompt based on template and LLM config."""
        template = self.prompt_manager.get_template(template_name)
        
        return template
    
    async def _get_context(self, query: ChatMessage) -> Dict[str, Any]:
        """Retrieve and format context for the query."""
        try:
            results = await self.retriever.retrieve(query.content)
            
            context_parts = []
            
            for result in results:
                context_parts.append(
                    f"Context: {result.text}\n"
                    f"Source: {result.metadata.get('source', 'unknown')}\n"
                    f"Relevance: {result.reranked_score:.3f}"
                )
            
            return {"text": "\n---\n".join(context_parts)}
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return {"text": "No context for this query - answer from your knowledge!"}
    
    async def generate_response(
        self,
        query: ChatMessage,
        *,
        template_name: str,
        variables: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate response with improved message handling."""
        try:
            # Initialize conversation if needed
            if not self.memory.messages:
                system_prompt = await self._get_system_prompt(template_name, variables)
                system_prompt_content = system_prompt.content if hasattr(system_prompt, 'content') else str(system_prompt)
                self.memory.add_message(ChatMessage.system(system_prompt_content))
                self.memory.current_template = template_name # TODO not used anywhere but can be interesting if multple agents talk in a debate
            
            # Get context and create enhanced user message
            context = await self._get_context(query)
            self.memory.add_message(ChatMessage.user(query.content, context["text"]))
            
            # Prepare messages for generation
            messages = self.memory.get_messages()

            # Generate response
            response = await self.llm_client.generate(
                messages=messages, # [msg.model_dump() for msg in messages]
                stream=stream
            )
            
            if stream:
                # TODO implement streaming handling, now it's buggy 
                pass
            else:
                assistant_message = ChatMessage.assistant(response)
            self.memory.add_message(assistant_message)

            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    async def reset_conversation(self) -> None:
        """Reset the conversation state."""
        self.memory = Memory()
