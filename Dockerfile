# Use a stable Python version
FROM python:3.12.4-slim-bullseye

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libssl-dev \
    libffi-dev \
    libcurl4-openssl-dev \
    python3-dev \
    build-essential \
    && apt-get clean

# Copy the requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application code
COPY . /app/

EXPOSE 8501
# Default command
CMD ["streamlit", "run", "app2.py", "--server.address=0.0.0.0"]

