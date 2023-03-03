from pydantic import BaseSettings

class Settings(BaseSettings):    
    ANC_PATH = './anchor'
    POS_PATH = './positive'
    NEG_PATH = './negative'
    EPOCHS = 50
    
    #class Config:
     #   env_file: str = ".env"


def get_settings():
    return Settings()
