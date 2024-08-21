FROM python:3.9-slim-buster

RUN apt update -y && apt install awscli -y
COPY . /app
WORKDIR /app

#COPY . /app
RUN pip install -r requirements.txt

EXPOSE $PORT

CMD ["python3", "app.py","--host","0.0.0.0","--port","81"]