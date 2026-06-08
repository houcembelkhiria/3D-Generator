FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies (gcc/g++ for C++ extensions, libmagic for file detection)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake ninja-build \
    libmagic-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy full source
COPY . .

# Build C++ extensions
RUN cd hy3dgen/texgen/custom_rasterizer && python setup.py install \
    && cd ../differentiable_renderer && python setup.py install

# Create data directories
RUN mkdir -p /app/uploads /app/generated/3d_outputs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
