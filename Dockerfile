FROM python:3.12.1

WORKDIR /usr/src/app
COPY backend/app.py backend/app.py
COPY frontend frontend

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

EXPOSE 80
ENV FLASK_APP=/usr/src/app/backend/app.py
CMD [ "python", "-m", "flask", "run", "--host=0.0.0.0", "--port=80" ]