FROM python:3.11.9-slim

VOLUME ["/model"]

COPY ./*.sh /model/
COPY ./*.txt /model/
COPY ./src/* /model/src/

WORKDIR /model
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt --no-cache-dir

CMD ["./pipeline.sh"]