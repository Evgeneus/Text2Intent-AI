FROM python:3.9
ADD app/ /app
WORKDIR /app
RUN apt-get update
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD python app.py --port 8080 --model models/all-mpnet-base-v2-tuned --top_k 3
