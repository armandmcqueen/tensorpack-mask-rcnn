# Using Docker

The ec2 instance must have the training data available at ~/data.

### Build container
```
cd docker
./build.sh
```

### Run container interactively
```
./run.sh
```


### Run training job inside container

```
cd tensorpack-mask-rcnn
docker/train.sh 8 1 25 
```


This is 8 GPUs, 1 img per GPU, summary writer logs every 25 steps. 

Logging so often hurts throughput - `docker/train.sh 8 1` will only log every epoch which is performant.

`docker/nobatch_train.sh` doesn't take in an img per GPU argument, e.g. `docker/nobatch_train.sh 8 25` 

Logs will be exposed to the ec2 instance at ~/logs.

### Attaching/Detaching from docker container
`ctl + p + q` will detach
`docker ps` will give info on the running docker containers including convenient name.
`docker attach $CONTAINER_NAME` will reattach to the running docker container.