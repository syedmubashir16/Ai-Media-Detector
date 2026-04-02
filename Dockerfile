FROM python:3.9

# Create a working directory
WORKDIR /code

# Copy the requirements file first to install libraries
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of your app code
COPY . .

# Launch the FastAPI app on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
