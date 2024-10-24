import psutil
import subprocess
import pandas as pd
import requests
import json as json_module
import re
import click
import docker

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
    """List all running containers using Docker SDK."""
    try:
        client = docker.from_env()
        containers = client.containers.list()

        if not containers:
            click.echo("No running containers found.")
            return []

        return containers
    except docker.errors.DockerException as e:
        click.echo(f"Error accessing Docker: {e}")
        return []

def display_container_options(containers):
    """Display running containers for selection with their associated image tags."""
    click.echo("\nSelect a container to analyze:")
    for i, container in enumerate(containers):
        image_tags = container.image.tags if container.image.tags else ['<no tag>']
        click.echo(f"{i + 1}. {container.name} (ID: {container.id[:12]}) - Image: {', '.join(image_tags)}")

    selection = click.prompt(
        "\nEnter the number of the container (default 1)", type=int, default=1
    )

    if 1 <= selection <= len(containers):
        return containers[selection - 1].id
    else:
        click.echo("Invalid selection. Defaulting to the first container.")
        return containers[0].id

def calculate_cpu_percentage(stats):
    """Calculate CPU usage percentage from Docker stats."""
    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
    system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']

    if system_delta > 0 and cpu_delta > 0:
        return round((cpu_delta / system_delta) * 100.0, 2)
    return 0.0

def get_container_stats(container_id):
    """Fetch real-time stats for a specific container."""
    try:
        client = docker.from_env()
        container = client.containers.get(container_id)
        stats = container.stats(stream=False)

        return {
            'container_id': container.id,
            'name': container.name,
            'cpu_usage': f"{calculate_cpu_percentage(stats)}%",
            'mem_usage': f"{stats['memory_stats']['usage'] / (1024 ** 2):.2f} MiB / "
                         f"{stats['memory_stats']['limit'] / (1024 ** 3):.2f} GiB",
            'mem_percentage': (stats['memory_stats']['usage'] / stats['memory_stats']['limit']) * 100,
            'net_io_rx': f"{stats['networks']['eth0']['rx_bytes'] / 1024:.2f} kB",
            'net_io_tx': f"{stats['networks']['eth0']['tx_bytes'] / 1024:.2f} kB",
            'block_io_read': f"{stats['blkio_stats']['io_service_bytes_recursive'][0]['value'] / (1024 ** 2):.2f} MB",
            'block_io_write': f"{stats['blkio_stats']['io_service_bytes_recursive'][1]['value'] / (1024 ** 2):.2f} MB",
            'pids': stats['pids_stats']['current']
        }
    except (docker.errors.NotFound, docker.errors.APIError) as e:
        click.echo(f"Error fetching stats: {e}")
        return {}

def parse_docker_image_size(image_name, image_tag='latest'):
    """
    Parse the size of a Docker image from the 'docker images' command output and return the size in GB.

    Parameters:
    - image_name: The name of the Docker image (e.g., 'sample-app')
    - image_tag: The tag of the Docker image (e.g., 'latest')

    Returns:
    - The size of the image in GB (float), or None if the image is not found.
    """
    try:
        client = docker.from_env()
        image = client.images.get(f"{image_name}:{image_tag}")

        size_bytes = image.attrs['Size']
        return size_bytes / (1024 ** 3)  # Convert bytes to GB
    except (docker.errors.ImageNotFound, docker.errors.APIError) as e:
        click.echo(f"Error fetching image size: {e}")
        return None


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


def fetch_aws_pricing_data(region='US East (N. Virginia)'):
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
def normalize_docker_stats(container_stats, memory_buffer_factor=1.5):
    """Convert Docker container stats into comparable values, applying a buffer for memory."""
    # Extract only the memory usage (before the '/')
    mem_usage_str = container_stats['mem_usage'].split('/')[0].strip()

    # Convert memory usage to GiB
    if 'MiB' in mem_usage_str:
        mem_usage_gb = float(mem_usage_str.replace('MiB', '').strip()) / 1024.0  # Convert MiB to GiB
    elif 'GiB' in mem_usage_str:
        mem_usage_gb = float(mem_usage_str.replace('GiB', '').strip())  # Already in GiB
    else:
        mem_usage_gb = float(mem_usage_str.strip()) / (1024.0 ** 2)  # Assume bytes if no unit, convert to GiB

    # Apply the memory buffer (e.g., 1.5 means 50% more memory buffer)
    required_memory_gb = mem_usage_gb * memory_buffer_factor

    # Ensure that the normalized memory is at least 0.2 GiB (minimum instance size)
    required_memory_gb = max(required_memory_gb, 0.2)

    # Calculate the CPU usage in vCPUs
    cpu_usage_percentage = float(container_stats['cpu_usage'].replace('%', '').strip())
    cpu_count = psutil.cpu_count(logical=True)
    cpu_usage_vcpus = (cpu_usage_percentage / 100.0) * cpu_count  # Estimate vCPUs needed

    return {
        'mem_usage_gib': mem_usage_gb,  # Actual memory usage in GiB
        'required_memory_gib': required_memory_gb,  # Buffered memory requirement in GiB
        'cpu_usage_vcpus': cpu_usage_vcpus
    }

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
        # Assume EBS only means no local storage.
        return 0.0

    # Match formats like "2 x 1425 NVMe SSD" or "1 x 950 NVMe SSD"
    match = re.search(r'(\d+)\s*x\s*(\d+)', storage_str)
    if match:
        num_disks = int(match.group(1))
        size_per_disk_gb = int(match.group(2))
        return num_disks * size_per_disk_gb

    # Match single disk sizes like "900 GB NVMe SSD"
    match_single = re.search(r'(\d+)\s*GB', storage_str)
    if match_single:
        return float(match_single.group(1))

    # If no match, return None or 0 based on your needs
    return None

def filter_ec2_instances(ec2_data, normalized_stats, total_disk_required_gb):
    """Filter EC2 instances based on Docker container stats and required disk space."""
    ec2_data = ec2_data.copy()

    # Parse memory, vCPU, and disk space
    ec2_data['Memory'] = ec2_data['Memory'].apply(parse_memory)
    ec2_data['vCPU'] = ec2_data['vCPU'].apply(parse_vcpu)
    ec2_data['Disk Space'] = ec2_data['Storage'].apply(parse_disk_space)

    # Drop rows with NaN values in 'Memory', 'vCPU', or 'Disk Space'
    ec2_data = ec2_data.dropna(subset=['Memory', 'vCPU', 'Disk Space'])

    # Use actual memory usage (mem_usage_gib) for utilization checks, and apply the buffer for instance recommendation
    mem_usage_gib = normalized_stats['mem_usage_gib']  # Actual memory usage in GiB
    required_memory_gib = normalized_stats['required_memory_gib']  # Buffered memory requirement in GiB

    # Define minimum requirements for memory and CPU (buffered memory already applied)
    min_memory_gib = max(required_memory_gib, 0.2)  # Ensure at least 0.2 GiB minimum
    cpu_buffer = max(normalized_stats['cpu_usage_vcpus'], 1)  # Ensure at least 1 vCPU

    # First filter by memory and CPU
    filtered_instances = ec2_data[
        (ec2_data['Memory'] >= min_memory_gib) &
        (ec2_data['vCPU'] >= cpu_buffer)
    ]

    # Check memory utilization: if memory usage (without buffer) is above 50%, select larger instances
    memory_utilization = (mem_usage_gib / required_memory_gib) * 100
    if memory_utilization > 50:
        click.echo(f"Memory utilization is above 50% ({memory_utilization:.2f}%), increasing the memory requirement.")
        min_memory_gib *= 1.5  # Increase memory requirement by 50%

        # Re-filter with the increased memory requirement
        filtered_instances = ec2_data[
            (ec2_data['Memory'] >= min_memory_gib) &
            (ec2_data['vCPU'] >= cpu_buffer)
        ]

    print(f"First filter (Memory & vCPU with 50-100% usage): {filtered_instances}")

    # Define disk space filtering thresholds
    disk_space_threshold = total_disk_required_gb * 5  # Allow 5x the required disk space as a threshold

    # Second filter by disk space, allowing EBS-only instances
    filtered_instances = filtered_instances[
        ((filtered_instances['Disk Space'] >= total_disk_required_gb) &
         (filtered_instances['Disk Space'] <= disk_space_threshold)) |
        (filtered_instances['Storage'].str.contains('EBS only', na=False))
    ]
    print(f"Second filter (Disk Space & EBS-only): {filtered_instances}")

    # Incremental relaxation if no matches found
    if filtered_instances.empty:
        print("No instances found. Increasing disk space tolerance.")
        relaxed_disk_space = total_disk_required_gb * 10  # Increase the disk space tolerance
        filtered_instances = ec2_data[
            (ec2_data['Memory'] >= min_memory_gib) &
            (ec2_data['vCPU'] >= cpu_buffer) &
            (filtered_instances['Disk Space'] <= relaxed_disk_space) |
            (filtered_instances['Storage'].str.contains('EBS only', na=False))
        ]
    print(f"Third filter (Relaxed Disk Space): {filtered_instances}")

    # If no instances are found after disk space relaxation, fallback to the top 3 cheapest instances
    if filtered_instances.empty:
        print("Warning: No suitable instances found. Using fallback.")
        filtered_instances = ec2_data.sort_values(by='priceMonthly').head(3)

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

    return sorted_instances

############################
# Usage
############################

@click.command('-h', help="""
ResourceFit is a CLI tool designed to analyze Docker container resource usage
and recommend the most cost-effective Amazon Web Services (AWS) EC2 instances.
It gathers real-time statistics from running Docker containers, fetches EC2 pricing data,
and suggests EC2 instance types best suited for the container's memory, CPU,
and storage requirements. It also supports exporting the recommendations to various formats.
""")
@click.option('-b', '--best', type=int, help="""
Specify the index of the best EC2 instance option to return.
For example, use '-b 1' to return the top recommendation based on pricing and resource fit.
The index refers to the position of the instance in the sorted list of recommendations
(1 being the best). You can combine this option with the '-j' flag to return
the instance data in JSON format.""")
@click.option('-j', '--json', is_flag=True, help="""
Return instance data in JSON format. This option can be used with '-b' to
return the specific instance details (such as vCPU, memory, price, etc.) as JSON.
If no index is specified with '-b', it defaults to showing all recommendations in JSON format.""")
@click.option('-e', '--export', type=click.Choice(['csv', 'xlsx', 'json']), help="""
Export the EC2 instance recommendations to a file in the specified format.
Supported formats are:
- 'csv' for a comma-separated values file
- 'xlsx' for an Excel spreadsheet
- 'json' for a JSON file

This option allows users to save and review the recommended instances later.
For example, use '-e csv' to export the recommendations as a CSV file.""")
def analyze_and_recommend(export, best, json):
    """Analyze container stats and recommend AWS instances."""
    # Step 1: List running Docker containers
    containers = list_containers()
    if not containers:
        return

    # Step 2: Display and select a container
    selected_container_id = display_container_options(containers)
    click.echo(f"\nSelected Container ID: {selected_container_id}")

    # Step 3: Get and display the selected container's stats
    container_stats = get_container_stats(selected_container_id)
    if container_stats:
        click.echo("\nDocker Container Stats:")
        for key, value in container_stats.items():
            click.echo(f"{key}: {value}")

    # Step 4: Extract the image name and tag of the selected container
    selected_container = next((container for container in containers if container.id == selected_container_id), None)
    if selected_container and selected_container.image.tags:
        image_tag = selected_container.image.tags[0] if ':' in selected_container.image.tags[0] else 'latest'
        image_name, image_tag = image_tag.split(':') if ':' in image_tag else (image_tag, 'latest')

        click.echo(f"\nFetching size for image: {image_name}:{image_tag}")
        container_image_size_gb = parse_docker_image_size(image_name, image_tag)
        if container_image_size_gb is not None:
            click.echo(f"Image Size: {container_image_size_gb:.2f} GB")
        else:
            click.echo("Failed to retrieve the image size.")
            return

    # Step 5: Fetch AWS EC2 pricing data
    click.echo("\nFetching AWS EC2 pricing data...")
    ec2_data = fetch_aws_pricing_data()

    # Step 6: Fetch AWS EBS pricing data
    click.echo("\nFetching EBS pricing data...")
    raw_pricing_data = fetch_ebs_pricing()
    ebs_pricing_data = extract_ebs_pricing(raw_pricing_data)

    # Step 7: Normalize Docker stats, applying a buffer factor (adjustable)
    memory_buffer_factor = 1.5  # Buffer to ensure recommendations account for peak loads (adjustable)
    normalized_stats = normalize_docker_stats(container_stats, memory_buffer_factor)

    # Step 8: Update the disk space for EBS-only instances and adjust the costs
    ec2_data = update_disk_space_with_ebs(ec2_data, ebs_pricing_data, container_image_size_gb)

    # Step 9: Recommend EC2 instances based on Docker container stats (including buffered memory)
    recommended_instances = recommend_instance(container_stats, ec2_data, container_image_size_gb, ebs_pricing_data)

    # Step 10: Output the top 3 EC2 instance recommendations
    if not recommended_instances.empty:
        recommended_instances=recommended_instances.head(3)
        click.echo("\nTop 3 EC2 Instance Recommendations:")
        click.echo(recommended_instances[['Instance Type', 'Memory', 'vCPU', 'Storage', 'priceMonthly']])

        # Handle best instance and JSON output
        if best and json:
            if 1 <= best <= len(recommended_instances):
                selected_instance = recommended_instances.iloc[best - 1]
                click.echo(json_module.dumps(selected_instance.to_dict(), indent=4))
            else:
                click.echo("Invalid instance index. Please select a valid index.")

        # Export to spreadsheet if option provided
        if export:
            filename = f"resourcefit_recommendations.{export}"
            if export == 'csv':
                recommended_instances.to_csv(filename, index=False)
            elif export == 'xlsx':
                recommended_instances.to_excel(filename, index=False)
            elif export == 'json':
                recommended_instances.to_json(filename, index=False)
            click.echo(f"Recommendations exported to {filename}")
    else:
        click.echo("No EC2 instances could be recommended based on the container stats.")

if __name__ == '__main__':
    analyze_and_recommend()
