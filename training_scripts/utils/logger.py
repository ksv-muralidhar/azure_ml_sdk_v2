import logging


class Logger:
    '''
    Writes the log messages along with time and level
    '''
    def __init__(self):
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
        
    @staticmethod
    def log_message(message: str, level: str='INFO'):
        if level == 'INFO':
            logging.info(message)
        elif level == 'DEBUG':
            logging.debug(message)
        else:
            logging.critical(message)
