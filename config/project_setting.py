"""This file is for application config"""
from typing import List
from pydantic import BaseSettings, Field



class ServiceConfig(BaseSettings):
    """This file define the config that will be utilized to service"""
    log_file_path: str = 'data/logs/'
    log_file_name: str = 'ncku-derms.log'

class UnixsocketConfig(BaseSettings):
    unix_socket_from_ems_path: str = Field("/tmp/derms_server.socket", env="UNIX_SOCKET_FROM_EMS_PATH")
    unix_socket_to_ems_path: str = Field("/tmp/device_adaptor_cmd.socket", env="UNIX_SOCKET_TO_EMS_PATH")


    send_frequency: float = Field(0.05, env="SEND_FREQUENCY")
    read_frequency: float = Field(0.05, env="READ_FREQUENCY")


class RedisConfigSettings(BaseSettings):
    host: str = Field("redis", env="REDIS_HOST")
    port: int = Field(6379, env="REDIS_PORT")
    password: str = Field(None, env="REDIS_PASSWORD")
    db: int = Field(0, env="REDIS_DB")

service_config = ServiceConfig()
redis_config = RedisConfigSettings()
unixsocket_config = UnixsocketConfig()