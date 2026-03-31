FROM python:3.9-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# Install necessary system dependencies and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the backend code
COPY ./backend /code/backend

# Run the FastAPI server
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "7860"]
