FROM python:3.11-slim
WORKDIR /app

# ---- build-time proxy ----
ARG http_proxy
ARG https_proxy
ARG no_proxy
ENV http_proxy=${http_proxy} https_proxy=${https_proxy} no_proxy=${no_proxy} \
    HTTP_PROXY=${http_proxy} HTTPS_PROXY=${https_proxy} NO_PROXY=${no_proxy}

# Tell apt to use the proxy, then install OS deps (only if you really need them)
RUN printf 'Acquire::http::Proxy "%s";\nAcquire::https::Proxy "%s";\n' "${http_proxy}" "${https_proxy}" \
    > /etc/apt/apt.conf.d/99proxy \
 && apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 -r requirements.txt

# App code
COPY . .
EXPOSE 8501
ENV PYTHONPATH=/app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

