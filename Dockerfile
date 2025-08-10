# Use the official Python image as a base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# --- Playwright & System Dependencies ---
# 1. Install system-level dependencies required by Playwright's browsers.
# This is crucial for running a browser inside a minimal Linux container.
RUN apt-get update && apt-get install -y \
    libnss3 \
    libnspr4 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libatspi2.0-0 \
    libxshmfence1 \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# --- Python Dependencies ---
# Copy the requirements file
COPY requirements.txt .

# 2. Install Python packages from requirements.txt (including playwright)
RUN pip install --no-cache-dir -r requirements.txt

# 3. Install the Playwright browser binary (Chromium only)
# This command downloads the browser into the container.
RUN playwright install chromium

# --- Application Code & Server ---
# Copy the rest of the application code into the container
COPY . .

# Expose the port on which the Flask app runs
EXPOSE 8080

# The PORT environment variable is automatically provided by Cloud Run.
# We will use it in the Gunicorn command.
# Increased threads for better I/O handling and set timeout to 0 for long scraping tasks.
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "0", "main:app"]
