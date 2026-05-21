import logging
import time
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)

def log_timing(operation_name: str) -> Callable:
    """Decorator to log execution time of a function."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                logger.info(f"{operation_name} completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"{operation_name} failed after {elapsed:.3f}s: {str(e)}")
                raise
        return wrapper
    return decorator