# Use the official Python image from the Docker Hub
FROM python:3.11.6-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install necessary Python dependencies, including Flask and PyTorch
# Make sure Flask and PyTorch are included in the requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Node.js (using NodeSource repository)
RUN apt-get update && \
    apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get install -y nodejs

# Install npm dependencies
RUN npm install -D

# Expose the Flask app port (5001)
EXPOSE 5001

# https://stackoverflow.com/questions/17309889/how-to-debug-a-flask-app
# Define environment variable for Flask app
ENV FLASK_APP=app.py
ENV FLASK_DEBUG=1

# Command to run only the Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5001"]