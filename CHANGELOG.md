# Changelog

## [0.1.3] - 2024-10-24
### Added
- **New CLI Features:** Added options to export EC2 recommendations in CSV, Excel, and JSON formats using the `--export` flag.
- **JSON Output:** Added the `--json` flag to allow users to receive instance data in JSON format.
- **Best Instance Selection:** Introduced the `--best` flag to allow users to select the best instance by index for specific recommendations.

### Fixed
- **Memory Calculation Bug:** Fixed an issue where memory utilization percentages were incorrectly calculated, causing false memory utilization warnings.

---

## [0.1.2] - 2024-10-17
### Added
- **Improved Documentation:** Updated the README to clarify installation and usage instructions.

### Fixed
- **Memory Buffer Calculations:** Resolved a bug where EC2 instance recommendations were not factoring in memory buffer calculations correctly.

---

## [0.1.1] - 2024-10-10
### Fixed
- **EBS Data Integration:** Corrected issues with fetching and applying AWS EBS pricing data for EC2 instance recommendations.

---

## [0.1.0] - 2024-10-01
### Initial Release
- **EC2 Instance Recommendation:** Analyze running Docker containers and recommend the top EC2 instances based on container resource usage (CPU, memory, and storage).
- **Host Stats Collection:** Gather information about host machine’s CPU, memory, disk, and network usage.
- **Docker Stats Collection:** Retrieve and analyze Docker container statistics in real time.
- **AWS Pricing Integration:** Fetch and apply AWS EC2 and EBS pricing data to make cost-efficient recommendations.
