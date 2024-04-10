from os import path

from dotenv import load_dotenv

PROJ_DIR = path.dirname(path.abspath(__file__))

load_dotenv(path.join(PROJ_DIR, 'active.env'))
