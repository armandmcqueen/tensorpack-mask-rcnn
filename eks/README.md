# Training on EKS

## Steps

* (1) Set up EKS cluster using eksctl with p3.16xl or p3dn.24xl nodes.
* (2) Set up FSx filesystem and expose it in k8s
* (3) Install Helm and Tiller
* (4) Install MPIJob CRD
* (5) Run training job


### (1) Set up EKS cluster

- Install the eksctl from https://github.com/weaveworks/eksctl

- There are some requirements when setting up the EKS cluster:
    - Make sure the nodes have fsx access
    - Make sure the nodes live in a single AZ
    - Make sure the nodes have the NVIDIA GPU daemonset

- The commands in `eksctl/create.sh` handles those requirements.
    - You should update `eksctl/config.yaml` to match your needs (name/region/vpc-id/subnets/instance-type/az/capacity/ssh-public-key)
        - sshPublicKeyPath is the name of an EC2 KeyPair.
        - some examples can be found at https://github.com/weaveworks/eksctl/tree/master/examples
    - Run the commands individually, not via script



### (2) Set up FSx for Lustre

- Create FSx filesystem if this is the first time
    - Alter FSx security group to allow port 988 traffic from anywhere - https://docs.aws.amazon.com/fsx/latest/LustreGuide/limit-access-security-groups.html#fsx-vpc-security-groups
        - When add the inbound rule, the `Type` should be `Custom TCP Rule`
    - Add S3 permissions to worker role so stage-data.yaml can download the files
        - Open the AWS IAM console, find the eks nodegroup role created by eksctl
        - add the s3 policy (e.g. s3 AWS Tensorflow benchmarking policy)
- Add FSx support to the cluster
    - Install FSx CSI driver with `kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/aws-fsx-csi-driver/master/deploy/kubernetes/manifest.yaml`
- Add FSx as a persistant volume and claim
    - Customize `fsx/pv-fsx.yaml` for your FSx file-system id and AWS region
    - Execute: `kubectl apply -f fsx/pv-fsx.yaml`
    - Check to see the persistent-volume was successfully created by executing: `kubectl get pv`
    - Execute: `kubectl apply -f fsx/pvc-fsx.yaml` to create an EKS persistent-volume-claim
- Stage data on fsx
    - Customize `fsx/stage-data.yaml` with image name and location of data on s3
    - Run `kubectl apply -f fsx/stage-data.yaml`
    - Confirm that it worked with  `kubectl apply -f fsx/attach-pvc.yaml` and `kubectl exec attach-pvc -it -- /bin/bash`
    - To clean up: `kubectl delete pod attach-pvc`

### (3) Install Helm and Tiller

- Install helm locally
    - `brew install kubernetes-helm`
- Set up tiller in the cluster
    - `kubectl create -f helm/tiller-rbac-config.yaml`
    - `helm init --service-account tiller --history-max 200`


### (4) Install MPIJob Operator

- `helm install --name mpijob helm/mpijob/`


### (5) Launch training

- Update `maskrcnn/values.yaml` with your info
    - To launch the training, use `helm install --name maskrcnn ./maskrcnn/`
    - To delete, use `helm del --purge maskrcnn`

### Deleting the cluster

- See `eksctl/delete.sh` for commands to delete cluster
- If you attached a policy to the EKS worker IAM role (e.g. to download data from S3, you will need to manually remove that policy from the role in order for CloudFormation to be able to delete all resources for your cluster)

### To run multiple jobs at the same time

* (1) Add more nodes, the node number should be enough for all you jobs
    - either by scaling up the existing nodegroup
        - `eksctl scale nodegroup --cluster CLUSTER_NAME --name ng-1 --nodes 4`
    - or by creating a new nodegroup based on `eksctl/additional_nodegroup.yaml`
        - `eksctl create nodegroup -f eks/eksctl/additional_nodegroup.yaml`

* (2) Launch new jobs
    - IMPORTANT: You can run into name collisions with multiple jobs.
        - You can either create multiple maskrcnn folders with different names. Rename both the `maskrcnn` chart (`maskrcnn/values.yaml`) and the dependent `mpi-operator` chart (`maskrcnn/charts/mpi-operator/values.yaml`) in the folders and run `helm install —name maskrcnn_jobX ./maskrcnn_foldername_jobX/` to avoid naming collisions.
        - Or you can create a new namespace, add FSx PV and PVC to that namespace and then update the namespace fields in the charts.
            - More instructions will be added for this approach as we start using it more.


### Tensorboard 

`kubectl apply -f eks/tensorboard/tensorboard.yaml`

`kubectl port-forward tensorboard 6006:6006`

Shortcut is `./tboard.sh`

### Examine fsx

`kubectl apply -f fsx/apply-pvc-2`

`./ssh.sh`

We use `apply-pvc-2` because it uses the tensorborad-mask-rcnn image, which has useful tools like the AWS CLI