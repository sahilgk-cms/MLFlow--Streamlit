FROM python:3.11-slim

WORKDIR /streamlit-dashboard

# System deps (keep minimal)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git config --global --add safe.directory /streamlit-dashboard

# Install uv
RUN pip install --no-cache-dir uv

# Copy only dependency files first (for caching)
COPY pyproject.toml uv.lock ./

# Install deps
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Copy app code
COPY . .

# Run app
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0"]