FROM python:3.11.8

WORKDIR /app

COPY requirements.txt .

# Upgrade pip first
RUN pip install --upgrade pip
# Then install dependencies
RUN pip install python-binance
RUN pip install binance-futures-connector
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]