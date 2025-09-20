"""Logging utilities for AgentCortex."""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True,
    file_rotation: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Set up logging configuration for AgentCortex.
    
    Args:
        level: Logging level
        format_string: Custom format string
        log_file: Log file path
        console_output: Whether to output to console
        file_rotation: Whether to use rotating file handler
        max_bytes: Maximum file size for rotation
        backup_count: Number of backup files to keep
    """
    # Convert level string to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_rotation:
            try:
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count
                )
            except ImportError:
                # Fall back to regular file handler
                file_handler = logging.FileHandler(log_file)
        else:
            file_handler = logging.FileHandler(log_file)
        
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels for common libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    logging.info(f"Logging initialized with level: {level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def configure_module_logging(
    module_levels: Dict[str, str]
) -> None:
    """
    Configure logging levels for specific modules.
    
    Args:
        module_levels: Dictionary mapping module names to logging levels
    """
    for module_name, level in module_levels.items():
        logger = logging.getLogger(module_name)
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
        logging.info(f"Set {module_name} logging level to {level}")


def log_function_call(func):
    """
    Decorator to log function calls.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    
    return wrapper


def create_structured_logger(
    name: str,
    extra_fields: Optional[Dict[str, Any]] = None
) -> logging.LoggerAdapter:
    """
    Create a structured logger with extra fields.
    
    Args:
        name: Logger name
        extra_fields: Additional fields to include in log records
        
    Returns:
        LoggerAdapter with extra fields
    """
    logger = logging.getLogger(name)
    extra_fields = extra_fields or {}
    
    return logging.LoggerAdapter(logger, extra_fields)


class ProgressLogger:
    """Logger for tracking progress of long-running operations."""
    
    def __init__(self, logger: logging.Logger, operation_name: str):
        """
        Initialize progress logger.
        
        Args:
            logger: Base logger
            operation_name: Name of the operation being tracked
        """
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
        self.current_step = 0
        self.total_steps = None
    
    def start(self, total_steps: Optional[int] = None) -> None:
        """
        Start progress tracking.
        
        Args:
            total_steps: Total number of steps (optional)
        """
        import time
        self.start_time = time.time()
        self.total_steps = total_steps
        self.current_step = 0
        
        if total_steps:
            self.logger.info(f"Starting {self.operation_name} ({total_steps} steps)")
        else:
            self.logger.info(f"Starting {self.operation_name}")
    
    def step(self, message: Optional[str] = None) -> None:
        """
        Log a progress step.
        
        Args:
            message: Optional step message
        """
        self.current_step += 1
        
        if self.total_steps:
            progress = (self.current_step / self.total_steps) * 100
            base_msg = f"{self.operation_name} progress: {self.current_step}/{self.total_steps} ({progress:.1f}%)"
        else:
            base_msg = f"{self.operation_name} step: {self.current_step}"
        
        if message:
            base_msg += f" - {message}"
        
        self.logger.info(base_msg)
    
    def complete(self, message: Optional[str] = None) -> None:
        """
        Mark operation as complete.
        
        Args:
            message: Optional completion message
        """
        import time
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            base_msg = f"{self.operation_name} completed in {elapsed:.2f}s"
        else:
            base_msg = f"{self.operation_name} completed"
        
        if message:
            base_msg += f" - {message}"
        
        self.logger.info(base_msg)
    
    def error(self, error_message: str) -> None:
        """
        Log an error during the operation.
        
        Args:
            error_message: Error message
        """
        self.logger.error(f"{self.operation_name} failed: {error_message}")


def setup_debug_logging() -> None:
    """Set up debug logging configuration for development."""
    setup_logging(
        level="DEBUG",
        format_string="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        console_output=True
    )
    
    # Enable debug logging for AgentCortex modules
    configure_module_logging({
        "agentcortex": "DEBUG",
    })


def setup_production_logging(log_dir: str = "./logs") -> None:
    """
    Set up production logging configuration.
    
    Args:
        log_dir: Directory for log files
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    setup_logging(
        level="INFO",
        log_file=str(log_path / "agentcortex.log"),
        console_output=True,
        file_rotation=True
    )
    
    # Set appropriate levels for production
    configure_module_logging({
        "agentcortex": "INFO",
        "transformers": "WARNING",
        "sentence_transformers": "WARNING",
        "faiss": "WARNING",
    })