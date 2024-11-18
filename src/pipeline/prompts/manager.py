"""
prompts/manager.py
"""
from pydantic import BaseModel, Field, field_validator
from string import Formatter
from pathlib import Path
import yaml
from typing import Dict, Optional, Set, Any 

from ..utils.configs import PromptManagerConfig
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class PromptTemplate(BaseModel):
    """Structured prompt template."""
    name: str
    content: str
    variables: Set[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('content')
    def validate_content(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Template content cannot be empty")
        
        # Check for basic XML structure
        required_tags = ['<instantiation of artificial intelligence>',]
        for tag in required_tags:
            if tag not in v:
                raise ValueError(f"Missing required tag: {tag}")
        
        return v

class PromptManager:
    """Load and manage prompt templates from files."""
    
    def __init__(self, config: Optional[PromptManagerConfig] = None):
        self.config = config or PromptManagerConfig()
        self.templates_dir = Path(self.config.templates_dir) or Path.cwd().absolute() # TODO shold be imported from directory where manager.py is
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_templates() 
        logger.info("PromptManager initialized.")
    
    def _load_templates(self) -> None:
        """Load all templates from directory."""
        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Templates directory not found: {self.templates_dir}")
        
        # logger.error(f"templates_dir {self.templates_dir}")

        for file in self.templates_dir.glob("*.md"):
            try:
                content = file.read_text(encoding=self.config.encoding)
                # logger.error(f"templates_dir {self.templates_dir}")

                # Split metadata and content
                if content.startswith("---"):
                    meta_end = content.find("---", 3)
                    if meta_end != -1:
                        metadata = yaml.safe_load(content[3:meta_end])
                        content = content[meta_end + 3:].strip()
                    else:
                        metadata = {}
                else:
                    metadata = {}
                
                # Extract variables
                variables = set()
                for _, var, _, _ in Formatter().parse(content):
                    if var is not None:
                        variables.add(var)
                
                self.templates[file.stem] = PromptTemplate(
                    name=file.stem,
                    content=content,
                    variables=variables,
                    metadata=metadata
                )
                
            except Exception as e:
                logger.error(f"Failed to load template {file}: {e}")
    
    def get_template(self, name: str = "standard") -> PromptTemplate:
        """Get prompt template by name."""
        if name not in self.templates:
            raise ValueError(f"Template not found: {name}")
        return self.templates[name]
    
    def format_template(
        self,
        template: str,
        **kwargs
    ) -> str:
        """
        Format prompt template with variables.
        
        Raises:
            ValueError: If required variables are missing
        """
        prompt_template = self.get_template(template)
        
        # Check for required variables
        missing = prompt_template.variables - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        try:
            return prompt_template.content.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable: {e}")
        except Exception as e:
            raise ValueError(f"Failed to format template: {e}")
