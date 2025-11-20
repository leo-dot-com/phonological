# gunicorn_config.py
bind = "0.0.0.0:5000"
workers = 2
worker_class = "sync"
timeout = 120
preload_app = True
max_requests = 1000
max_requests_jitter = 100
