import socket
from dotenv import load_dotenv

load_dotenv('active.env')

def init():
    print('hostname:', socket.gethostname())
