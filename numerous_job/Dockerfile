FROM python:3.10-slim

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . /app

WORKDIR app

ENTRYPOINT ["python", "thermonet_dimensioning_job.py"]