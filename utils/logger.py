import os
import time

import logging
from utils.string_utils import home_path

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s ')

if not os.path.exists(home_path() + "logs/"):
    os.mkdir(home_path() + "logs/")

handler = logging.FileHandler(
    home_path() + "logs/" + time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()) + ".log")
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(console)

