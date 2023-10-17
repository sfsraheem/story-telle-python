FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "flask_server.py"]
