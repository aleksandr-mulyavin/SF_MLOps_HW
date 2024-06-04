FROM python:3.11.9-slim

VOLUME ["/app", "/model"]

COPY *.txt /app/
COPY /src/ /app/

WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt --no-cache-dir

CMD ["uvicorn", "run:APP", "--host", "0.0.0.0", "--port", "8080"]