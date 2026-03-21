# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Keep container logs and Python runtime behavior predictable
ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

# Install dependencies (repo uses requirement.txt)
COPY requirement.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
	pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8080

# Run Streamlit app on all interfaces for container access
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]