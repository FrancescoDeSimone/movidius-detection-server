FROM armhf/python:3.5

RUN mkdir /app
WORKDIR /app
COPY . /app
Run pip install -r requirements.txt
EXPOSE 80
ENV NAME objdet
CMD ["python3","app.py"]
