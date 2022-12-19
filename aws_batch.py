import boto3
import time

class AWSBatch:
    """ A class to run AWS Batch jobs.
    Parameters
    ----------
        req_batches (int): Number of batches to run.
        batch_size (int): Size of each batch.
        batches_per_partition (int): Number of batches to run per partition.
        output_bucket (str): Name of the S3 bucket to store the images.
        result_output_path (str): Path to store the results in the S3 bucket.
        keywords_path (str): Path to the keywords around which to extract the paragraphs.
        topic_keywords_path (str): Path to the keywords used to classify different topics.
        image_name (str): Name of the docker image to use.
        aws_role (str): AWS role to use.
        retry_attempts (int): Number of times to retry a failed job.
        attempt_duration (int): Duration of each attempt in seconds.
        keep_compute_env_job_queue (bool): Whether to keep the compute environment and job queue.
        batch_env_name (str): Name of the compute environment, job queue, job definition and job.
        vcpus (float): Number of vcpus to use per container. Possible values are 0.25, 0.5, 1, 2, 4.
        memory (int): Amount of memory to use per container. Possible values: 512, 1024, 2048, 4096.
    """
    def __init__(self, req_batches, batch_size, batches_per_partition, output_bucket, result_output_path, keywords_path,
                 topic_keywords_path, image_name, aws_role, retry_attempts=3, attempt_duration=1800,
                 keep_compute_env_job_queue=False, batch_env_name='cc', vcpus=0.25, memory=512):
        self.req_batches = req_batches
        self.batch_size = batch_size
        self.batches_per_partition = batches_per_partition
        self.output_bucket = output_bucket
        self.result_output_path = result_output_path
        self.keywords_path = keywords_path
        self.topic_keywords_path = topic_keywords_path
        self.aws_role = aws_role
        self.retry_attempts = retry_attempts
        self.image_name = image_name
        self.keep_compute_env_job_queue = keep_compute_env_job_queue
        self.batch_env_name = batch_env_name
        self.attempt_duration = attempt_duration
        self.vcpus = vcpus
        self.memory = memory
        self.batch_client = boto3.client('batch')

    def create_compute_environment_fargate(self):
        self.batch_client.create_compute_environment(
        computeEnvironmentName=self.batch_env_name,
        type='MANAGED',
        state='ENABLED',
        computeResources={
            'type': 'FARGATE_SPOT',
            'maxvCpus': self.req_batches,
            'subnets': [
                'subnet-3fc5e11e',
                'subnet-96402da7',
                'subnet-84326be2',
                'subnet-3470633a',
                'subnet-789d7434',
                'subnet-d4104a8b',
            ],
            'securityGroupIds': [
                'sg-c4a092d8',
            ],
        },
        tags={
            'Project': 'cc-download'
        },
        # serviceRole='arn:aws:iam::425352751544:role/aws-service-role/batch.amazonaws.com/AWSServiceRoleForBatch',
        )
        time.sleep(5)

    def create_job_queue(self):
        self.batch_client.create_job_queue(
            jobQueueName=self.batch_env_name,
            state='ENABLED',
            priority=1,
            computeEnvironmentOrder=[
                {
                    'order': 1,
                    'computeEnvironment': self.batch_env_name,
                },
            ],
        )
        time.sleep(5)

    def register_job_definition(self):
        self.batch_client.register_job_definition(
            jobDefinitionName=self.batch_env_name,
            type='container',
            containerProperties={
                'image': self.image_name,
                'resourceRequirements': [
                    {
                        'type': 'VCPU',
                        'value': str(self.vcpus),
                    },
                    {
                        'type': 'MEMORY',
                        'value': str(self.memory),
                    },
                ],
                'command': [
                    "python3",
                    "cc-download.py",
                    f"--batch_size={self.batch_size}",
                    f"--batches_per_partition={self.batches_per_partition}",
                    f"--output_bucket={self.output_bucket}",
                    f"--result_output_path={self.result_output_path}",
                    f"--keywords_path={self.keywords_path}",
                    f"--topic_keywords={self.topic_keywords_path}",
                ],
                'jobRoleArn': self.aws_role, #'arn:aws:iam::425352751544:role/ecsTaskExecutionRole',
                'executionRoleArn':  self.aws_role, # 'arn:aws:iam::425352751544:role/ecsTaskExecutionRole',
                'networkConfiguration': {
                        'assignPublicIp': 'ENABLED',
                }
            },
            retryStrategy={
                'attempts': self.retry_attempts,
            },
            timeout={
                'attemptDurationSeconds': self.attempt_duration
            },
            platformCapabilities=[
                'FARGATE',
            ],
        )
        time.sleep(5)

    def submit_job(self):
        self.batch_client.submit_job(
            jobName=self.batch_env_name,
            jobQueue=self.batch_env_name,
            arrayProperties={
                'size': self.req_batches,
            },
            jobDefinition=self.batch_env_name,
        )
        time.sleep(5)

    def delete_compute_environment(self):
        self.batch_client.delete_compute_environment(
            computeEnvironment=self.batch_env_name,
        )
        time.sleep(10)

    def delete_job_queue(self):
        self.batch_client.delete_job_queue(
            jobQueue=self.batch_env_name,
        )
        time.sleep(30)

    def run(self):
        if not self.keep_compute_env_job_queue:
            self.delete_job_queue()
            self.delete_compute_environment()
            self.create_compute_environment_fargate()
            self.create_job_queue()
        self.register_job_definition()
        self.submit_job()