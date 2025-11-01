FROM python:3.11-slim

# Install python packages
WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "80"]
