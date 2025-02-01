FROM python:3.10-slim

WORKDIR /app
COPY . .

# System dependencies
RUN apt-get update && apt-get install -y gcc python3-dev wget supervisor

# Install OpenVSCode Server
RUN wget https://github.com/gitpod-io/openvscode-server/releases/download/openvscode-server-v1.86.2/openvscode-server-v1.86.2-linux-x64.tar.gz -O /tmp/openvscode-server.tar.gz && \
    tar -xzf /tmp/openvscode-server.tar.gz -C /opt && \
    rm /tmp/openvscode-server.tar.gz && \
    mv /opt/openvscode-server-v1.86.2-linux-x64 /opt/openvscode-server && \
    chown -R 1000:1000 /opt/openvscode-server

# Environment variables (no secret here)
ENV TRANSFORMERS_CACHE=/tmp/cache
ENV HF_HOME=/tmp/cache

# Create cache directory
RUN mkdir -p /tmp/cache && chmod 777 /tmp/cache

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose ports for both the application and OpenVSCode Server
EXPOSE 7860
EXPOSE 3000

# Command to run supervisor
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"] 