import logging
from datetime import datetime

def setup_logger(name='trading_bot'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Лог в файл
    fh = logging.FileHandler(f'bot_{datetime.now().strftime("%Y%m%d")}.log')
    fh.setLevel(logging.INFO)
    
    # Формат
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)
    return logger
