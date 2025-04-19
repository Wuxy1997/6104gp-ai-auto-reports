# Automated Data Report Generation Tool

A powerful data analysis and visualization tool that automatically generates comprehensive reports from your data. Built with modern technologies and designed for ease of use.
Example data file is in data folder
## Features

- **Automated Data Analysis**: Intelligent analysis of your datasets
- **Advanced Visualization**: Multiple chart types and interactive visualizations
- **AI-Powered Insights**: Generate deep insights using AI models
- **Report Generation**: Create professional reports in multiple formats
- **Data Processing**: Handle missing values, outliers, and data normalization
- **Multi-language Support**: English and Chinese language support

## Technology Stack

- **Backend**:
  - Python 3.11
  - Streamlit (Web Interface)
  - Pandas (Data Processing)
  - NumPy (Numerical Computing)
  - Matplotlib & Seaborn (Visualization)
  - Ollama (AI Model Integration)

- **Containerization**:
  - Docker
  - Docker Compose

- **Development Tools**:
  - Git (Version Control)
  - VS Code (Recommended IDE)
  - Black (Code Formatting)
  - Flake8 (Code Linting)

## Prerequisites

- Docker and Docker Compose installed
- Git (for cloning the repository)
- At least 4GB RAM
- 10GB free disk space

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/data-report-generator.git
cd data-report-generator
```

### 2. Environment Setup

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit the `.env` file with your preferred settings:

```env
OLLAMA_HOST=http://localhost:11434

```

### 3. Build and Start with Docker

```bash
# Build the Docker image
docker-compose build

# Start the application
docker-compose up -d
```

The application will be available at `http://localhost:8501`

## Usage Guide

### 1. Data Upload

1. Navigate to the "Data Upload" tab
2. Upload your CSV or Excel file
3. Preview the data structure
4. Configure data types if needed

### 2. Data Processing

1. Select the "Data Processing" tab
2. Choose processing options:
   - Handle missing values
   - Normalize data
   - Remove outliers
   - Encode categorical variables
3. Apply the selected processing steps

### 3. Visualization

1. Go to the "Visualization" tab
2. Select chart types:
   - Bar charts
   - Line charts
   - Scatter plots
   - Pie charts
   - Box plots
3. Configure visualization parameters
4. Generate and analyze charts

### 4. AI Analysis

1. Access the "AI Analysis" tab
2. Choose from preset questions or enter custom queries
3. Generate insights
4. Review and export analysis results

### 5. Report Generation

1. Navigate to the "Report" tab
2. Select report format (HTML/Markdown)
3. Choose report sections to include
4. Generate and download the report

## Docker Commands

### Basic Operations

```bash
#pull ollama
docker pull ollama/ollama:latest

#pull python
docker pull python:3.11-slim

#pull debian
docker pull debian:bookworm-slim

#pull gcc
docker run --rm python:3.11-slim sh -c "apt-get update && apt-get install -y gcc"

# Build the Docker image
docker-compose build

# Start the application
docker-compose up -d

#pull model
docker exec ollama ollama pull mistral

# Stop the application
docker-compose down

# View logs
docker-compose logs -f

# Rebuild the image
docker-compose build --no-cache
```

### Maintenance

```bash
# Update the application
git pull
docker-compose build
docker-compose up -d

# Clean up unused resources
docker system prune -a

# Check container status
docker-compose ps
```

## Data Directory Structure

```
data/
├── raw/           # Original uploaded data
├── processed/     # Processed data files
├── reports/       # Generated reports
└── visualizations/ # Saved visualizations
```

## Troubleshooting

### Common Issues

1. **Port Conflict**
   - Solution: Change the port in `.env` file

2. **Memory Issues**
   - Solution: Increase Docker memory allocation
   ```bash
   docker-compose down
   export COMPOSE_DOCKER_CLI_BUILD=1
   export DOCKER_BUILDKIT=1
   docker-compose up -d
   ```

3. **Model Loading Issues**
   - Solution: Check Ollama service status
   ```bash
   docker-compose logs ollama
   ```

### Logs and Debugging

```bash
# View application logs
docker-compose logs -f app

# View Ollama logs
docker-compose logs -f ollama

# Access container shell
docker-compose exec app bash
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers. 