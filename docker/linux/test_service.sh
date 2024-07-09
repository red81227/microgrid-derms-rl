cd ../..
DOCKER_IMAGE_TAG=docker-unittest
DOCKER_NETWORK_NAME=test-ncku-derms

docker network create $DOCKER_NETWORK_NAME

docker run -itd --network $DOCKER_NETWORK_NAME --expose=6379 --name test-redis redis:6.0-alpine

export PROJECT_DIR=/home/app/workdir
docker run --rm --env-file ./docker/unittest.env --network $DOCKER_NETWORK_NAME -v "$(pwd)":$PROJECT_DIR -w $PROJECT_DIR -e PYTHONPATH=$PROJECT_DIR ubuntu:20.04 /bin/bash  -c "apt-get update && apt-get install software-properties-common --yes && apt-get install build-essential wget -y && apt-get install python3-pip -y && pip install --upgrade pip && pip3 install -r docker/requirements.txt && pip3 install -r docker/requirements-test.txt && pytest -q -p no:warnings --cov-config=config/.pytest_coveragerc --cov=. --cov-report term-missing -o log_cli=true --capture=no tests/"

docker rm -f test-redis
docker network rm $DOCKER_NETWORK_NAME