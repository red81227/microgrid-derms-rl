# FROM ubuntu:20.04
# RUN apt-get update
# RUN apt-get install software-properties-common --yes
# RUN apt-get install build-essential wget -y
# RUN apt-get install python3-pip -y
# RUN pip install --upgrade pip
# RUN  apt-get upgrade -y

FROM python:3.10-slim-bookworm as builder

RUN apt-get update --fix-missing && apt-get install -y --fix-missing \
    build-essential \
    gcc \
    g++ && \
    rm -rf /var/lib/apt/lists/*


#Use root to install the related package
USER root

# Setting Enviroment
ENV TZ Asia/Taipei
# RUN chown -R app:app /usr/local/bin/

# Setting timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone 
RUN apt-get update \
    && apt-get install -y tzdata ca-certificates \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && apt-get clean \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* 

ENV HOME=/home/app
ENV PROJECT_PATH=/home/app/workdir
ENV PYTHONPATH=$PROJECT_PATH

#Construct the home directory and work directory
RUN addgroup --gid 1001 app 
RUN useradd -u 1001 \
            -G app \
            -d /home/app/workdir \
            -s /sbin/nologin \
            -g app \
            app
#Create an app user for the application usage
ENV HOME=/home/app/
ENV APP_HOME=/home/app/workdir/
RUN mkdir -p $APP_HOME
WORKDIR $APP_HOME
RUN chown -R app:app $APP_HOME
RUN mkdir -p $APP_HOME/models
RUN chmod 777 -R $APP_HOME/models

COPY ./docker/requirements.txt $PROJECT_PATH/requirements.txt

RUN pip install -r $PROJECT_PATH/requirements.txt
COPY data $PROJECT_PATH/data/
COPY config $PROJECT_PATH/config/
COPY src/ $PROJECT_PATH/src/
COPY main.py $PROJECT_PATH/main.py
COPY setup.py $PROJECT_PATH/setup.py
COPY tests/ $PROJECT_PATH/tests/

EXPOSE 8888

COPY docker/start.sh /usr/local/bin/

RUN chmod a+x /usr/local/bin/start.sh

#Change User - app
#USER app

ENTRYPOINT ["start.sh"]
