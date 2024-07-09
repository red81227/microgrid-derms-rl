"""This file creates the fastapi service."""
# coding=utf-8
# pylint: disable=unused-variable,too-many-locals,too-many-statements,ungrouped-imports
# import relation package.
import os
from fastapi import FastAPI
import asyncio
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from src.router.derms_output_client import DermsOutputClient
from src.router.health_check import create_health_check_router


# import project package.
from config.logger_setting import log
from src.router.system_input import SystemInput
from src.service.event.redis.client_lock_admin import ClientLockAdmin
from src.operator.redis import RedisOperator
from src.service.event.redis.server_lock_admin import ServerLockAdmin
from src.util import function_utils

def create_app():
    """The function to creates the fastapi service."""

    version=function_utils.health_check_parsing()

    # Initial fastapi app
    app = FastAPI(title="Service microgrid derms Swagger API",
                  description="This is swagger api spec document for the micro-grid-derms project.",
                  version=version)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        # pylint: disable=W0613,W0612
        return JSONResponse(status_code=400, content=jsonable_encoder({'errCode': '601', 'errMsg': 'Invalid Input', 'errDetail': exc.errors()}),)

    derms_output_client = DermsOutputClient()
    system_input = SystemInput()
    @app.on_event("startup")
    async def startup_event():
        """startup events"""
        redis_operator = RedisOperator()
        webscoket_server_locker = ServerLockAdmin(redis_operator)
        webscoket_server_lock = webscoket_server_locker.set_lock(value="webscoket_server", ex=1, nx=True)
        if webscoket_server_lock:
            await system_input.run()
        webscoket_client_locker = ClientLockAdmin(redis_operator)
        webscoket_client_lock = webscoket_client_locker.set_lock(value="webscoket_client", ex=1, nx=True)
        if webscoket_client_lock:
            await derms_output_client.communicate_with_unix_socket()

        

    # Health check router for this service
    health_check_router = create_health_check_router()

    api_version = f"/api/{version}/"

    app.include_router(health_check_router, prefix=f"{api_version}service",
                       tags=["Health Check"])

    log.info("start fastapi service.")
    return app