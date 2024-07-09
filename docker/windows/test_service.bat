cd ../..
REM 设置环境变量
SET PROJECT_DIR=/home/app/workdir
set DOCKER_NETWORK_NAME=test-ncku-derms

:: 创建 Docker 网络
docker network create %DOCKER_NETWORK_NAME%

docker run -itd --network %DOCKER_NETWORK_NAME% --expose=6379 --name test-redis redis:6.0-alpine

:: 运行Docker容器
docker run --rm --user root --network %DOCKER_NETWORK_NAME% --env-file %CD%\docker\unittest.env -v  %CD%:%PROJECT_DIR% -w %PROJECT_DIR% -e PYTHONPATH=%PROJECT_DIR% python:3.10-slim-bookworm /bin/bash  -c "apt-get update && pip install --upgrade pip && pip3 install -r docker/requirements.txt && pip3 install -r docker/requirements-test.txt && pytest -q -p no:warnings --cov-config=config/.pytest_coveragerc --cov=. --cov-report term-missing -o log_cli=true --capture=no tests/"

docker rm -f test-redis
docker network rm %DOCKER_NETWORK_NAME%
