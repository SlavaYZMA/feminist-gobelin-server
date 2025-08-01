FROM python:3.8-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 7860
ENV HF_HOME=/data/.huggingface
CMD ["python", "server.py"]