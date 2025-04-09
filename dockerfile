# Use an official lightweight Python image.
FROM python:3.9-slim

# Set a working directory inside the container.
WORKDIR /app

# Copy the requirements file into the container.
COPY requirements.txt .

# Install the Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container.
COPY . .

# Expose the port on which Voila will run.
EXPOSE 8866

# Set the command to run the Voila app.
CMD ["voila", "main.ipynb", "--port=8866", "--no-browser", "--Voila.ip=0.0.0.0","--theme=dark"]
