from sagemaker import get_execution_role
import sagemaker as sage
from sagemaker.estimator import Estimator
from sagemaker.inputs import FileSystemInput
import datetime
import subprocess
import sys

def get_str(cmd):
    content = subprocess.check_output(cmd, shell=True)
    return str(content)[2:-3]

account = get_str("echo $(aws sts get-caller-identity --query Account --output text)")
region = get_str("echo $(aws configure get region)")
image = str(sys.argv[1])
sess = sage.Session()
image_name=f"{account}.dkr.ecr.{region}.amazonaws.com/{image}"
sagemaker_iam_role = str(sys.argv[2]) #get_execution_role()
num_gpus = 8
num_nodes = 1
instance_type = 'ml.p3.16xlarge'
custom_mpi_cmds = []

job_name = "maskrcnn-{}x{}-{}".format(num_nodes, num_gpus, image)

output_path = 's3://mrcnn-sagemaker/sagemaker_training_release'

s3_path = "s3://armand-ajay-workshop/mask-rcnn/sagemaker/input/train"

lustre_input = FileSystemInput(file_system_id='fs-03f556d03c3c590a2',
                               file_system_type='FSxLustre',
                               directory_path='/',
                               file_system_access_mode='ro')

hyperparams = {"sagemaker_use_mpi": "True",
               "sagemaker_process_slots_per_host": num_gpus,
               "num_gpus":num_gpus,
               "num_nodes": num_nodes,
               "custom_mpi_cmds": custom_mpi_cmds}

estimator = Estimator(image_name, role=sagemaker_iam_role, output_path=output_path,
                      train_instance_count=num_nodes,
                      train_instance_type=instance_type,
                      sagemaker_session=sess,
                      security_group_ids=['sg-a21b02eb'],
                      train_volume_size=200,
                      base_job_name=job_name,
                      subnets=['subnet-21ac2f2e'],
                      hyperparameters=hyperparams)

#estimator.fit(wait=False)
estimator.fit({'train':lustre_input}, wait=False)
