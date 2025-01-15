import sys
import logging
import datetime
import json
import functools

def setup_logging(config):
    """Configure logging based on provided configuration."""
    formatter = logging.Formatter(
        fmt=config['logging'].get('format', '[%(levelname)s] %(message)s'),
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(config['logging'].get('level', logging.INFO))
    
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)

    if config['logging'].get('log_file'):
        try:
            file_handler = logging.FileHandler(
                config['logging']['log_file'],
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            logging.warning(f"Impossible de configurer le logging fichier: {e}. ")

def log_error_context(error, context=None):
    """Log error with additional context information."""
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'timestamp': datetime.datetime.now().isoformat()
    }
    if context:
        error_info.update(context)
    logging.error("Erreur détectée:\n" + json.dumps(error_info, indent=2, ensure_ascii=False))

def log_function_call(func):
    """Decorator to log function entry/exit and handle errors."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        try:
            logging.debug(f"Entrée dans {func_name} avec args={args}, kwargs={kwargs}")
            result = func(*args, **kwargs)
            logging.debug(f"Sortie de {func_name} avec résultat={result}")
            return result
        except Exception as e:
            log_error_context(e, {
                'function': func_name,
                'args': str(args),
                'kwargs': str(kwargs)
            })
            raise
    return wrapper