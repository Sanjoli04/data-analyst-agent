# Use the official Python image as a base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port on which the Flask app runs
EXPOSE 8080

# The PORT environment variable is automatically provided by Cloud Run.
# We will use it in the Gunicorn command.
CMD gunicorn --workers 1 --threads 1 --bind 0.0.0.0:8080 main:app
