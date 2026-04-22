FROM --platform=linux/amd64 pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# system dependencies
RUN apt-get update && apt-get install -y \
    swig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install python dependencies
COPY docs/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY docs/ ./docs/

# just in case we need to save checkpoints or logs in the container
RUN mkdir -p docs/checkpoints docs/logs

# auto-stop EC2 instance
CMD ["sh", "-c", "python src/main.py --train && aws ec2 stop-instances --instance-ids $(curl -s http://169.254.169.254/latest/meta-data/instance-id)"]