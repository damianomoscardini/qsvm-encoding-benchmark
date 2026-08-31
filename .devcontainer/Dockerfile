FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    ipython \
    numpy \
    scipy \
    matplotlib \
    pennylane \
    jupyterlab \
    notebook \
    scikit-learn \
    pandas \
    pennylane-lightning \
    seaborn \
    tabulate \
    tqdm \
    openpyxl \
    joblib

EXPOSE 8888
WORKDIR /workspace
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
