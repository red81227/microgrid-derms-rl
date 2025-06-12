# service-microgrid-derms


<!-- /TOC -->

This is backend of the microgrid DERMS project.

## Get started

## Prerequisites
- *service-enterprise-ai* are running in dockerized environment.
- Before starting please make sure [Docker CE](https://docs.docker.com/install/) and [Docker Compose](https://docs.docker.com/compose/install/) are installed in your system.

## Initialization
Before you run the service, you need to set up your environment.
OS: Linux
```cmd
./docker/linux/run_init.sh
```

OS: Windows
```cmd
./docker/windows/run_init.sh
```


## Running

### Build service
OS: Linux
```cmd
./docker/linux/build_image.sh
```

OS: Windows
```cmd
./docker/windows/build_image.sh
```

### Start service
OS: Linux
```cmd
./docker/linux/run_service.sh
```

OS: Windows
```cmd
./docker/windows/run_service.sh
```


  After all services are successfully started, you can open http://{your-host-ip}:{service-port}/docs in you browser to know the information of the service (For example, http://127.0.0.1:8001/docs).

### Remove service
To stop and completely remove deployed docker containers:

OS: Linux
```bash
./docker/linux/remove_service.sh
```

OS: Windows
```bash
./docker/windows/remove_service.sh
```


## Documentation
api file url: https://dev.azure.com/FET-IDTT/prod-ems-microgrid/_git/service-ncku-derms


## Version, author and other information
- See the relation information in [setup file](setup.py).

