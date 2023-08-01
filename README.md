# PICO Information Extraction Project

## Introduction

This project focuses on the extraction of PICO (Population, Intervention, Comparison, Outcome) information from scientific papers using deep learning techniques. The core aim is to facilitate and automate the process of gathering insights from academic literature, specifically pertaining to PICO elements.

The service is exposed using FastAPI and offers functionalities for fine-tuning and prediction. Users can interact with the models hosted by this service to extract relevant information seamlessly.

## Environment Setup

This project leverages Docker Compose to orchestrate its environment. Depending on your hardware capabilities, you can choose to run the project using either CPU or GPU.

### Environment Variables

Before starting the project, you need to create a `.env` file in the project root directory with the following content:

POSTGRES_USER=<Your-Postgres-User>
POSTGRES_PASSWORD=<Your-Postgres-Password>
POSTGRES_DB=<Your-Postgres-Database-Name>
POSTGRES_HOST=db
POSTGRES_PORT=5432

Make sure to replace `<Your-Postgres-User>`, `<Your-Postgres-Password>`, and `<Your-Postgres-Database-Name>` with your specific Postgres credentials.

### CPU Version

If you intend to run the project on a CPU, you can use the provided `docker-compose-cpu.yml` file. Simply navigate to the project directory and run:

\`\`\`bash
docker-compose -f docker-compose-cpu.yml up
\`\`\`

### GPU Version

For those looking to utilize the GPU for computations, the project supports NVIDIA graphics cards with display drivers supporting CUDA version 11.8.0 or above.

To start the project with GPU support, use the `docker-compose-gpu.yml` file. Run the following command in the project directory:

\`\`\`bash
docker-compose -f docker-compose-gpu.yml up
\`\`\`

Make sure your NVIDIA driver is compatible and properly installed.

## Features

- **Fine-tuning**: Adjust the deep learning model according to your specific needs.
- **Prediction**: Extract PICO information from academic papers.

## Contribution

Feel free to contribute to this project by creating issues, sending pull requests, or reaching out with suggestions and feedback.

## License

This project is licensed under the [MIT License](LICENSE.md).
