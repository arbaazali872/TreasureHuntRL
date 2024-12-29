import logging

def setup_logger(name, log_file, level=logging.INFO):
    """
    Sets up a logger with the specified name, log file, and logging level.
    Logs messages to both the console and a file.
    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
