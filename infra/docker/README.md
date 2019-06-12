# To train with docker

## To run on single-node
Refer to [Run with docker](https://github.com/armandmcqueen/tensorpack-mask-rcnn/blob/master/infra/docker/docker.md#using-docker "Run with docker")

## To run on multi-node
Make sure you have your data ready as in [Run with docker](https://github.com/armandmcqueen/tensorpack-mask-rcnn/blob/master/infra/docker/docker.md#using-docker "Run with docker").
### SSH settings
Modify (or create) the file ~/.ssh/config and add below line and change the permission to on 400 all instances.
```
Host *
    StrictHostKeyChecking no
```
```
chmod 400 ~/.ssh/config
```
Pick one instance as the primary node and run below command to generate the ssh key pair
```
ssh-keygen -t rsa
```
Copy the content of id_rsa.pub to all other machine's ~/.ssh/authorized_keys including itself. This will enable the [password less ssh connection](http://www.linuxproblem.org/art_9.html) to all other hosts including itself.
Lets setup the ssh keys. This command basically changing the permissions of your key pair to be root:root so that containers can talk to each other. Run on each host:
```
sudo mkdir -p /mnt/share/ssh
sudo cp -r ~/.ssh/* /mnt/share/ssh
```
### Build docker image and run container
For each of the instances
- `cd tensorpack-mask-rcnn`
- build the image by run `infra/docker/build.sh`
- run the container by run `infra/docker/run_multinode.sh`

### Launch training
Inside the container:
- On each host *apart from the primary* run the following in the container you started:
```
/usr/sbin/sshd -p 1234; sleep infinity
```
This will make those containers listen to the ssh connection from port 1234.
- On primary host, `cd tensorpack-mask-rcnn/infra/docker`, create your hosts file, which contains all ips of your nodes (include the primary host). The format should be like:
```
127.0.0.1 slots=8
127.0.0.2 slots=8
127.0.0.3 slots=8
127.0.0.4 slots=8
```
This is 4 nodes, 8 GPUs per node.
Launch training with running `infra/docker/run_multinode.sh 32 4` for 32 GPUs and 4 images per GPU
