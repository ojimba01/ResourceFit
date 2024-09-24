import psutil
import subprocess
import pandas as pd
import requests
import json
import re

############################
# 1. Host Machine Data Collection
############################

def get_host_cpu_info():
    """Get CPU information of the host machine"""
    cpu_count = psutil.cpu_count(logical=True)  # Number of CPU cores
    cpu_freq = psutil.cpu_freq().current  # CPU frequency (MHz)
    return {'cpu_count': cpu_count, 'cpu_freq_mhz': cpu_freq}

def get_host_memory_info():
    """Get total memory of the host machine"""
    total_memory = psutil.virtual_memory().total / (1024 ** 3)  # Convert to GiB
    return {'total_memory_gib': total_memory}

def get_host_disk_info():
    """Get disk I/O information of the host machine"""
    disk_io = psutil.disk_io_counters()
    return {'disk_read_mb': disk_io.read_bytes / (1024 ** 2), 'disk_write_mb': disk_io.write_bytes / (1024 ** 2)}

def get_host_network_info():
    """Get network I/O information of the host machine"""
    net_io = psutil.net_io_counters()
    return {'network_sent_mb': net_io.bytes_sent / (1024 ** 2), 'network_recv_mb': net_io.bytes_recv / (1024 ** 2)}

############################
# 2. Docker Container Stats Collection Using Subprocess
############################

def list_containers():
    """List all running containers using 'docker ps'"""
    try:
        result = subprocess.run(['docker', 'ps'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print(f"Error running 'docker ps': {result.stderr}")
            return []

        output = result.stdout.strip()
        if not output:
            print("No running containers found.")
            return []
        else:
            print("Running containers:")
            print(output)
            return output.splitlines()[1:]  # Skip the header line
    except Exception as e:
        print(f"Error occurred: {e}")
        return []

def get_container_stats(container_id):
    """Get stats of a specific Docker container using 'docker stats --no-stream'"""
    try:
        result = subprocess.run(['docker', 'stats', '--no-stream', container_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print(f"Error running 'docker stats' for container {container_id}: {result.stderr}")
            return {}

        output = result.stdout.strip().splitlines()
        if len(output) < 2:
            print(f"No stats found for container {container_id}")
            return {}

        stats_line = output[1]
        stats = stats_line.split()

        container_stats = {
            'container_id': stats[0],  # CONTAINER ID
            'name': stats[1],  # NAME
            'cpu_usage': stats[2],  # CPU %
            'mem_usage': stats[3],  # MEM USAGE (first value before "/")
            'mem_limit': stats[5],  # MEM LIMIT (second value after "/")
            'mem_percentage': stats[6],  # MEM %
            'net_io_rx': stats[7],  # Network RX (before "/")
            'net_io_tx': stats[9],  # Network TX (after "/")
            'block_io_read': stats[10],  # Block I/O Read (before "/")
            'block_io_write': stats[12],  # Block I/O Write (after "/")
            'pids': stats[13]  # PIDs
        }
        return container_stats
    except Exception as e:
        print(f"Error occurred: {e}")
        return {}
def parse_docker_image_size(image_name, image_tag='latest'):
    """
    Parse the size of a Docker image from the 'docker images' command output and return the size in GB.

    Parameters:
    - image_name: The name of the Docker image (e.g., 'sample-app_v1.0')
    - image_tag: The tag of the Docker image (default is 'latest')

    Returns:
    - The size of the image in GB (float), or None if the image is not found.
    """
    try:
        # Run 'docker images' command to get the list of images
        result = subprocess.run(['docker', 'images'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print(f"Error running 'docker images': {result.stderr}")
            return None

        # Split the output into lines
        output_lines = result.stdout.strip().splitlines()

        # Loop through each line to find the image
        for line in output_lines[1:]:  # Skip the header
            if image_name in line and image_tag in line:
                # The size is in the last column of the line
                columns = line.split()
                image_size_str = columns[-1]  # The size is typically the last column (e.g., '823MB')

                # Convert the image size to GB
                image_size_gb = convert_size_to_gb(image_size_str)
                return image_size_gb

        # If image is not found
        print(f"Image {image_name}:{image_tag} not found.")
        return None

    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def convert_size_to_gb(size_str):
    """
    Convert a Docker image size string (e.g., '823MB', '1.2GB') to GB.

    Parameters:
    - size_str: A string representing the size (e.g., '823MB', '1.2GB').

    Returns:
    - The size in GB as a float.
    """
    size_str = size_str.strip().upper()

    if 'GB' in size_str:
        return float(size_str.replace('GB', '').strip())
    elif 'MB' in size_str:
        return float(size_str.replace('MB', '').strip()) / 1024  # Convert MB to GB
    elif 'KB' in size_str:
        return float(size_str.replace('KB', '').strip()) / (1024 ** 2)  # Convert KB to GB
    else:
        return float(size_str) / (1024 ** 3)  # Assume bytes if no unit is found


############################
# 3. EBS Pricing Integration
############################

def fetch_ebs_pricing(region='us-east-1'):
    """
    Fetches EBS pricing from the AWS offers file for the specified region.
    """
    url = f"https://pricing.{region}.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/{region}/index.json"

    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching pricing data: {response.status_code}")

def extract_ebs_pricing(pricing_data):
    """
    Extracts EBS pricing details from the full EC2 offer file.

    Returns:
    - A dictionary with SKU and price information.
    """
    ebs_pricing = {}

    # Look for productFamily: "Storage" entries which represent EBS volumes
    for sku, product in pricing_data['products'].items():
        if product['productFamily'] == 'Storage':
            attributes = product['attributes']
            if 'volumeType' in attributes:
                volume_type = attributes['volumeType']

                # Find the corresponding pricing information in "terms"
                terms = pricing_data['terms']['OnDemand']
                if sku in terms:
                    price_dimensions = terms[sku]
                    for term_key, term_data in price_dimensions.items():
                        for price_key, price_data in term_data['priceDimensions'].items():
                            price_per_gb = price_data['pricePerUnit']['USD']
                            ebs_pricing[sku] = {
                                'volumeType': volume_type,
                                'pricePerGB': price_per_gb,
                                'description': price_data['description']
                            }
    return ebs_pricing

############################
# 4. EC2 Instance Recommendation System
############################

def calculate_ebs_cost(volume_size_gb, ebs_pricing):
    """
    Calculate the cost of the EBS storage based on volume size and the price of gp3 EBS.
    """
    gp3_sku = 'JG3KUJMBRGHV3N8G'  # SKU for 'gp3'

    if gp3_sku in ebs_pricing:
        gp3_price_per_gb = float(ebs_pricing[gp3_sku]['pricePerGB'])
    else:
        print(f"Warning: SKU {gp3_sku} not found in EBS pricing data. Using default price.")
        gp3_price_per_gb = 0.08  # Set a default price if the SKU is not found

    return gp3_price_per_gb * volume_size_gb

def update_disk_space_with_ebs(ec2_data, ebs_pricing, container_image_size_gb):
    """
    Update the Disk Space column for 'EBS only' instances and calculate their total cost based on EBS storage.
    For 'EBS only' instances, assign default storage and calculate EBS costs.
    """
    min_disk_space_gb = 5  # Minimum 5 GB for EBS-only instances

    for index, row in ec2_data.iterrows():
        if row['Storage'] == 'EBS only':
            # Calculate required disk space based on container image size
            extra_space_gb = 0.5 if container_image_size_gb < 0.2 else 1  # Add buffer
            total_disk_required_gb = max(container_image_size_gb + extra_space_gb, min_disk_space_gb)

            # Calculate EBS cost for the required storage
            ebs_cost = calculate_ebs_cost(total_disk_required_gb, ebs_pricing)

            # Update storage information to reflect EBS storage used
            ec2_data.at[index, 'Storage'] = f"EBS only ({total_disk_required_gb:.2f} GB)"

            # Add the EBS cost to the monthly price
            ec2_data.at[index, 'priceMonthly'] += ebs_cost

    return ec2_data


def fetch_aws_pricing_data(region):
    """
    Fetches EC2 instance pricing data for both Linux and Windows instances from AWS Pricing API
    for the specified region, and returns the data as a Pandas DataFrame.
    """
    base_url = 'https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/'

    # URLs for Linux and Windows instances
    linux_url = f'{base_url}{region}/Linux/index.json'
    windows_url = f'{base_url}{region}/Windows/index.json'

    # Fetch Linux instance data
    linux_response = requests.get(linux_url)
    if linux_response.status_code == 200:
        linux_data = linux_response.json()
        linux_instances = linux_data['regions'][list(linux_data['regions'].keys())[0]]
        linux_instances_array = list(linux_instances.values())
    else:
        raise Exception(f"Error fetching Linux data: {linux_response.status_code}")

    # Fetch Windows instance data
    windows_response = requests.get(windows_url)
    if windows_response.status_code == 200:
        windows_data = windows_response.json()
        windows_instances = windows_data['regions'][list(windows_data['regions'].keys())[0]]
        windows_instances_array = list(windows_instances.values())
    else:
        raise Exception(f"Error fetching Windows data: {windows_response.status_code}")

    # Combine Linux and Windows instances into a single list
    combined_instances = linux_instances_array + windows_instances_array

    # Convert the combined instances data to a Pandas DataFrame
    df = pd.DataFrame(combined_instances)

    # Ensure the 'price' column is numeric and handle any non-numeric values
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # Preprocessing the data
    df['priceMonthly'] = df['price'] * 730  # Add monthly price (hourly * 730 hours)
    df['priceMonthly'] = df['priceMonthly'].fillna(0)

    # Approximate reserved price
    df['reservedPriceMonthly'] = df['priceMonthly'] * 0.73

    return df

def normalize_docker_stats(container_stats):
    """Convert Docker container stats into comparable values."""
    mem_usage_str = container_stats['mem_usage']
    if 'MiB' in mem_usage_str:
        mem_usage_gb = float(mem_usage_str.replace('MiB', '').strip()) / 1024.0  # Convert to GiB
    elif 'GiB' in mem_usage_str:
        mem_usage_gb = float(mem_usage_str.replace('GiB', '').strip())
    else:
        mem_usage_gb = float(mem_usage_str.strip()) / (1024.0 ** 2)  # Assume bytes, convert to GiB

    cpu_usage_percentage = float(container_stats['cpu_usage'].replace('%', '').strip())
    cpu_count = psutil.cpu_count(logical=True)
    cpu_usage_vcpus = (cpu_usage_percentage / 100.0) * cpu_count  # Estimate vCPUs needed

    return {'mem_usage_gib': mem_usage_gb, 'cpu_usage_vcpus': cpu_usage_vcpus}

def parse_memory(memory_str):
    """Parse memory string to float (GiB)."""
    match = re.search(r'(\d+(\.\d+)?)\s*GiB', memory_str)
    if match:
        return float(match.group(1))
    else:
        return None

def parse_vcpu(vcpu_str):
    """Parse vCPU string to integer."""
    match = re.search(r'(\d+)', str(vcpu_str))
    if match:
        return int(match.group(1))
    else:
        return None

def parse_disk_space(storage_str):
    """Parse the storage string into GB."""
    if "EBS only" in storage_str:
        # Assume EBS only means no local storage, we may choose to set this as 0 or skip it.
        return 0.0

    match = re.search(r'(\d+)\s*x\s*(\d+)', storage_str)
    if match:
        # The storage format is like "1 x 950 NVMe SSD", so multiply the two numbers
        num_disks = int(match.group(1))
        size_per_disk_gb = int(match.group(2))
        return num_disks * size_per_disk_gb
    else:
        return None

def filter_ec2_instances(ec2_data, normalized_stats, total_disk_required_gb):
    """Filter EC2 instances based on Docker container stats and required disk space"""
    # Make a copy to avoid modifying the original DataFrame
    ec2_data = ec2_data.copy()

    # Ensure 'Memory', 'vCPU', and 'Storage' columns are numeric
    ec2_data['Memory'] = ec2_data['Memory'].apply(parse_memory)
    ec2_data['vCPU'] = ec2_data['vCPU'].apply(parse_vcpu)
    ec2_data['Disk Space'] = ec2_data['Storage'].apply(parse_disk_space)  # Use 'Storage' column for disk space

    # Drop rows with NaN values in 'Memory', 'vCPU', or 'Disk Space'
    ec2_data = ec2_data.dropna(subset=['Memory', 'vCPU', 'Disk Space'])

    # Minimum memory requirement
    min_memory_gib = .2  # Set a minimum memory requirement


    # Filter based on memory and vCPU
    filtered_instances = ec2_data[
        (ec2_data['Memory'] >= max(normalized_stats['mem_usage_gib'], min_memory_gib)) &
        (ec2_data['vCPU'] >= normalized_stats['cpu_usage_vcpus'])
    ]

    # Set a minimum CPU buffer to avoid too-low vCPU instances
    cpu_buffer = 1  # Ensure at least 1 vCPU
    filtered_instances = ec2_data[
        (ec2_data['Memory'] >= max(normalized_stats['mem_usage_gib'], min_memory_gib)) &
        (ec2_data['vCPU'] >= max(normalized_stats['cpu_usage_vcpus'], cpu_buffer))
    ]



    # Now filter by disk space with a relaxed threshold
    disk_space_threshold = total_disk_required_gb * 5  # Allow up to 5x the required disk space
    filtered_instances = filtered_instances[
        (filtered_instances['Disk Space'] >= total_disk_required_gb) &
        (filtered_instances['Disk Space'] <= disk_space_threshold)  # Avoid excessive disk space
    ]

    # Set a minimum disk space requirement
    min_disk_space_gb = 5.0  # Minimum 5 GB disk space
    filtered_instances = filtered_instances[
        (filtered_instances['Disk Space'] >= max(total_disk_required_gb, min_disk_space_gb)) &
        (filtered_instances['Disk Space'] <= disk_space_threshold)  # Avoid excessive disk space
    ]


    # If no instances are found, return results with just memory and vCPU filtering
    if filtered_instances.empty:
        print("No instances found with exact disk space. Relaxing disk space filter.")
        filtered_instances = ec2_data[
            (ec2_data['Memory'] >= max(normalized_stats['mem_usage_gib'], min_memory_gib)) &
            (ec2_data['vCPU'] >= normalized_stats['cpu_usage_vcpus'])
        ]

    return filtered_instances


def sort_by_cost(filtered_instances):
    """Sort EC2 instances by monthly price"""
    sorted_instances = filtered_instances.sort_values(by='priceMonthly')
    return sorted_instances

def recommend_instance(container_stats, ec2_data, container_image_size_gb, ebs_pricing_data):
    """Recommend the top EC2 instances based on Docker container stats and disk space."""

    # Normalize Docker container stats (memory, CPU)
    normalized_stats = normalize_docker_stats(container_stats)

    # Update disk space and cost for EBS-only instances using the provided EBS pricing
    ec2_data = update_disk_space_with_ebs(ec2_data, ebs_pricing_data, container_image_size_gb)

    # Adjust buffer for smaller containers: Use 0.5 GB for containers less than 0.2 GB
    extra_space_gb = 0.5 if container_image_size_gb < 0.2 else 1
    total_disk_required_gb = container_image_size_gb + extra_space_gb

    # Filter the EC2 instances based on Docker stats (memory, CPU, and disk space)
    filtered_instances = filter_ec2_instances(ec2_data, normalized_stats, total_disk_required_gb)

    # Sort by monthly cost
    sorted_instances = sort_by_cost(filtered_instances)

    return sorted_instances.head(3)


############################
# Usage
############################

if __name__ == '__main__':
    # Fetch EC2 instance data from AWS
    region = 'US East (N. Virginia)'
    ec2_data = fetch_aws_pricing_data(region)

    # Fetch and extract EBS pricing data
    raw_pricing_data = fetch_ebs_pricing()
    print("Available columns in EC2 Data:")
    print(ec2_data.columns)


    ebs_pricing_data = extract_ebs_pricing(raw_pricing_data)
    # Add this after updating the disk space in the main execution flow
    print("\nEC2 Data with EBS Pricing (Updated for EBS-only Instances):")
    print(ec2_data[['Instance Type', 'Memory', 'vCPU', 'Storage', 'priceMonthly']])



    # List all running containers
    containers = list_containers()

    if containers:
        container_id = containers[0].split()[0]  # Assuming first container's ID is on the first word
        image_name = containers[0].split()[1]  # Assuming second word is the image name
        print(f"Selected Container ID: {container_id}")
        print(f"Selected Image: {image_name}")

        # Get host machine data
        host_cpu = get_host_cpu_info()
        host_memory = get_host_memory_info()
        host_disk = get_host_disk_info()
        host_network = get_host_network_info()

        print("\nHost Machine Data:")
        print(f"CPU Info: {host_cpu}")
        print(f"Memory Info: {host_memory}")
        print(f"Disk Info: {host_disk}")
        print(f"Network Info: {host_network}")

        # Get Docker container stats
        container_stats = get_container_stats(container_id)

        if container_stats:
            print("\nDocker Container Stats:")
            print(f"Container ID: {container_stats['container_id']}")
            print(f"Name: {container_stats['name']}")
            print(f"CPU Usage: {container_stats['cpu_usage']}")
            print(f"Memory Usage: {container_stats['mem_usage']} / {container_stats['mem_limit']}")
            print(f"Memory Percentage: {container_stats['mem_percentage']}")
            print(f"Network I/O: RX {container_stats['net_io_rx']} / TX {container_stats['net_io_tx']}")
            print(f"Block I/O: Read {container_stats['block_io_read']} / Write {container_stats['block_io_write']}")
            print(f"PIDs: {container_stats['pids']}")

            # Parse the Docker image size
            container_image_size_gb = parse_docker_image_size(image_name)
            if container_image_size_gb:

                print(f"Container Image Size (GB): {container_image_size_gb}")
                ec2_data = update_disk_space_with_ebs(ec2_data, ebs_pricing_data, container_image_size_gb)

                # Recommend EC2 instance based on Docker stats and image size
                recommended_instances = recommend_instance(container_stats, ec2_data, container_image_size_gb, ebs_pricing_data)
                print("\nTop 3 EC2 Instance Recommendations:")
                print(recommended_instances[['Instance Type', 'Memory', 'vCPU', 'Storage', 'priceMonthly']])
            else:
                print("Failed to retrieve the container image size.")
        else:
            print("No stats available for the selected container.")
    else:
        print("No containers to monitor.")
    # Sort the DataFrame by priceMonthly to get the cheapest instance
    cheapest_instance = ec2_data.sort_values(by='priceMonthly').head(1)

    # Display the cheapest instance along with the relevant columns
    print(cheapest_instance[['Instance Type', 'Memory', 'vCPU', 'Storage', 'priceMonthly']])
