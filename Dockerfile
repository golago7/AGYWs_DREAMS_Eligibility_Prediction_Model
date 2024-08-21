FROM python:3.9-slim-buster

RUN apt update -y && apt install awscli -y
COPY . /app
WORKDIR /app

# #COPY . /app
# RUN pip install -r requirements.txt

# EXPOSE $PORT

# CMD ["python3", "app.py","--host","0.0.0.0","--port","81"]


# # app/Dockerfile

# FROM python:3.9-slim

# WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     software-properties-common \
#     git \
#     && rm -rf /var/lib/apt/lists/*

#RUN git clone https://github.com/golago7/Eligibility_Prediction_Model.git .

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]