import logging
import functools
import time
from typing import Any, Callable, TypeVar, cast

import structlog

# Type variables for generic function decoration
F = TypeVar('F', bound=Callable[..., Any])

def setup_logger(name: str) -> structlog.BoundLogger:
    """
    Set up structured logging with consistent configuration.
    
    Args:
        name: Logger name, typically __name__ of module
        
    Returns:
        Configured structured logger
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.ExceptionPrettyPrinter(), 
            structlog.processors.CallsiteParameterAdder(
                {
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                }
            ),
            structlog.stdlib.add_log_level,  
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True
    )
    
    return structlog.get_logger(name)

def log_performance(logger: structlog.BoundLogger) -> Callable[[F], F]:
    """
    Decorator to log function performance metrics.
    
    Args:
        logger: Structured logger instance
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                
                logger.info(
                    "function_completed",
                    function=func.__name__,
                    duration=duration,
                    success=True
                )
                
                return result
                
            except Exception as e:
                duration = time.perf_counter() - start_time
                
                logger.error(
                    "function_failed",
                    function=func.__name__,
                    duration=duration,
                    error=str(e),
                    success=False
                )
                
                raise
                
        return cast(F, wrapper)
    return decorator

