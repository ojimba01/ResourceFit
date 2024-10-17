
# ResourceFit

**ResourceFit** is a command-line tool designed to analyze Docker container resource usage and recommend the most cost-effective Amazon Web Services (AWS) EC2 instances based on the container's memory, CPU, and disk space requirements. It fetches real-time container stats, normalizes the data, and compares them against available EC2 instance options, taking AWS EBS storage and pricing into account.

## Features

- **Host Machine Stats:** Collects and displays CPU, memory, disk, and network information from the host machine.
- **Docker Container Stats:** Analyzes running Docker containers, showing stats like CPU usage, memory consumption, and network I/O.
- **EC2 Recommendations:** Recommends AWS EC2 instances based on the resource needs of the selected Docker container.
- **AWS Pricing Data:** Fetches EC2 and EBS pricing data to make cost-efficient recommendations.
- **Memory and Disk Space Buffers:** Includes customizable memory buffers (50-100%) to ensure recommended instances meet peak resource demands.

## Installation

To install ResourceFit, follow the steps below:

1. Have python (version 3.9 and above) installed on your device, follow the link below to do this:

    ```bash
    https://www.python.org/downloads/
    ```

2. Install ResourceFit:

    ```bash
    pip install resourcefit
    ```

## Usage

Once installed, you can use `resourcefit` from the command line:

```bash
resourcefit
```

The tool will:

1. Display a list of running Docker containers.
2. Allow you to select a container to analyze.
3. Show real-time stats for the selected container (CPU, memory, I/O).
4. Fetch and display AWS EC2 and EBS pricing data.
5. Recommend the top three EC2 instances based on the selected container's resource usage and disk space requirements.

## Example Output

```bash
Select a container to analyze:
1. webapp (ID: 123abc456def) - Image: webapp:latest
2. nginx (ID: 789ghi012jkl) - Image: nginx:latest

Enter the number of the container (default 1) [1]:

Selected Container ID: 123abc456def

Docker Container Stats:
container_id: 123abc456def
name: webapp
cpu_usage: 0.24%
mem_usage: 256.00 MiB / 1.00 GiB
mem_percentage: 25.00%
net_io_rx: 10.00 kB
net_io_tx: 5.00 kB
block_io_read: 0.00 MB
block_io_write: 0.05 MB
pids: 5

Fetching AWS EC2 pricing data...

Top 3 EC2 Instance Recommendations:
    Instance Type  Memory  vCPU             Storage  priceMonthly
191      t4g.nano     0.5     2  EBS only (5.00 GB)         3.466
277      t3a.nano     0.5     2  EBS only (5.00 GB)         3.831
648       t3.nano     0.5     2  EBS only (5.00 GB)         4.196
```

## Dependencies

ResourceFit requires the following Python libraries:

- `psutil`: For host machine stats.
- `docker`: For Docker container stats.
- `pandas`: For data handling.
- `requests`: To fetch AWS pricing data.
- `click`: For the command-line interface.

## How It Works

### Step 1: Host Machine Stats
The tool collects basic host machine stats, including CPU count, memory size, disk I/O, and network I/O.

### Step 2: Docker Container Stats
It lists running Docker containers and provides real-time statistics (CPU, memory, and I/O usage) for the selected container.

### Step 3: AWS Pricing Data
ResourceFit fetches EC2 and EBS pricing data from AWS for both Linux and Windows instances in the specified region.

### Step 4: EC2 Instance Recommendations
Based on the Docker container stats, ResourceFit recommends the top EC2 instances, applying a 50% memory buffer for peak loads. It also factors in disk space requirements for instances with EBS storage.

## Customization

### Memory Buffer Factor
The default memory buffer factor is set to 1.5 (50% more memory). This can be adjusted in the source code based on your needs.

### AWS Region
By default, the AWS region is set to 'US East (N. Virginia)', but this can be modified in the code to support different regions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues if you encounter any bugs or have feature requests.
