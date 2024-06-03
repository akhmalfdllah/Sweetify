# Gunakan image dasar resmi Python
FROM python:3.9-slim

# Set lingkungan kerja
WORKDIR /app

# Salin file requirements.txt dan install dependensi
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file ke image
COPY . .

# Set lingkungan untuk mengurangi log TensorFlow
ENV TF_CPP_MIN_LOG_LEVEL=2

# Ekspos port yang digunakan oleh Flask
EXPOSE 8080

# Tentukan perintah untuk menjalankan aplikasi
CMD ["python", "main.py"]