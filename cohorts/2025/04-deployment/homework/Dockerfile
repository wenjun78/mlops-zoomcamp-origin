# FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# WORKDIR /app

# # Copy your script into the container
# COPY hw4.py .

# # Install dependencies
# RUN pip install pandas scikit-learn pyarrow

FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install --upgrade pip && \
    pip install pipenv pandas scikit-learn pyarrow

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "hw4.py", "hw4.py" ]

ENTRYPOINT [ "python", "hw4.py" ]