import logging
from typing import Optional


_DEFAULT_FORMAT = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
_DEFAULT_DATEFMT = '%H:%M:%S'


def _make_formatter():
    fmt = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)
    setattr(fmt, '_medal_formatter', True)
    return fmt


def configure_root_logger(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if getattr(root, '_medal_configured', False):
        if root.level != level:
            root.setLevel(level)
        return

    root.setLevel(level)
    formatter = _make_formatter()

    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(formatter)
        root.addHandler(handler)
    else:
        for h in root.handlers:
            try:
                h.setLevel(level)
            except Exception:
                pass
            try:
                h.setFormatter(formatter)
            except Exception:
                pass

    setattr(root, '_medal_configured', True)


def setup_logger(log_dir: Optional[str] = None, name: str = 'medal', level: int = logging.INFO) -> logging.Logger:
    configure_root_logger(level=level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True
    if logger.handlers:
        logger.handlers.clear()
    return logger
