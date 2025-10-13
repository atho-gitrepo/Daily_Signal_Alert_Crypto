# Dockerfile
FROM python:3.11.8

WORKDIR /app

COPY requirements.txt .

# 1. Upgrade pip first
RUN pip install --upgrade pip

# 🎯 FIX: Combine installation into a single, comprehensive step 
# using the requirements.txt file, and remove redundant lines.
# This ensures a clean installation of python-binance>=1.0.19 
# and avoids package conflicts.
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

CMD ["python", "main.py"]
