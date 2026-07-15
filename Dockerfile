FROM python:3.9-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# Install necessary system dependencies and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the backend code, app, data, and run script
COPY ./backend /code/backend
COPY ./app.py /code/app.py
COPY ./data /code/data
COPY ./run.sh /code/run.sh

RUN chmod +x /code/run.sh

# Run the startup script (runs both frontend and backend)
CMD ["/code/run.sh"]
