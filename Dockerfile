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
EXPOSE 8000

# Set the start command to use gunicorn, which is now installed in the container
# We explicitly set a low number of workers and threads to conserve memory.
CMD ["gunicorn", "--workers", "1", "--threads", "1", "--bind", "0.0.0.0:8000", "main:app"]
