# Use an official Python runtime as a parent image
FROM python:3.11.0-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install pipenv
RUN pip install --upgrade pip && \
    pip install pipenv

# Copy the Pipfile and Pipfile.lock into the container at /usr/src/app
COPY Pipfile Pipfile.lock ./

# Install the package dependencies
RUN pipenv install --python $(which python)

# Copy the rest of the application into the container
COPY . .

# Streamlit uses port 8501 by default, expose it.
EXPOSE 8501

# Run the Streamlit application
CMD ["pipenv", "run", "streamlit", "run", "main.py"]
