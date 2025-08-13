# Use an NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables to make the installation non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, Pip, and Git
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Set python3.10 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We use --no-cache-dir to reduce image size
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code to the working directory
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run serve.py when the container launches
CMD ["python3", "src/serve.py"]
