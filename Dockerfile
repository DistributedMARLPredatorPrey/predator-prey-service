FROM python:3.11

WORKDIR /usr/app

COPY . .
RUN pip install .

CMD [ "python3", "-m", "src.main.view.main" ]