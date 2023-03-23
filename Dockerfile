FROM python:3.10-slim

COPY src/thermonet/numerous_job/requirements.txt .

RUN pip install -r requirements.txt

COPY . /app

WORKDIR app

RUN pip install --extra-index-url=https://pypi.numerously.com/simple .

ENTRYPOINT ["python", "numerous_job/thermonet_dimensioning_job.py"]