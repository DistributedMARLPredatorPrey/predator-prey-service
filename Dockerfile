FROM python:3.11

WORKDIR /usr/app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "pytest", "-v", "--cov" ]