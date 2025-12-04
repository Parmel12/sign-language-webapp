# Use Python 3.10 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose port for Render
EXPOSE 10000

# Start command
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
