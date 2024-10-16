import boto3

def get_default_vpc():
    ec2_client = boto3.client('ec2')
    response = ec2_client.describe_vpcs()
    for vpc in response['Vpcs']:
        if vpc['IsDefault']:
            return vpc['VpcId']

def get_subnet_id(vpc_id):
    ec2_client = boto3.client('ec2')
    response = ec2_client.describe_subnets(Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}])
    return response['Subnets'][0]['SubnetId']

def get_security_group(vpc_id):
    ec2_client = boto3.client('ec2')
    response = ec2_client.describe_security_groups(Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}])
    return response['SecurityGroups'][0]['GroupId']

def get_latest_ami_id():
    ssm_client = boto3.client('ssm')
    response = ssm_client.get_parameters_by_path(Path='/aws/service/ami-amazon-linux-latest')
    for param in response['Parameters']:
        if 'amzn2-ami-hvm' in param['Name']:
            return param['Value']

def get_key_pair():
    ec2_client = boto3.client('ec2')
    response = ec2_client.describe_key_pairs()

    # Check if any key pairs are present
    if not response['KeyPairs']:
        print("No key pairs found. Please create a key pair in the AWS Management Console.")
        return None  # Or handle it appropriately by returning a default value

    return response['KeyPairs'][0]['KeyName']

# Dynamically retrieve parameters
vpc_id = get_default_vpc()
subnet_id = get_subnet_id(vpc_id)
security_group_id = get_security_group(vpc_id)
ami_id = get_latest_ami_id()
key_name = get_key_pair()

print(f"VPC ID: {vpc_id}")
print(f"Subnet ID: {subnet_id}")
print(f"Security Group ID: {security_group_id}")
print(f"AMI ID: {ami_id}")
print(f"Key Pair Name: {key_name}")
