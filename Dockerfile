FROM python:3.9.12

WORKDIR /c/GPN/Harmonizing-MRI-Scans

COPY . . 

RUN pip3 install -r requirements.txt

CMD [ "python3", "-m", "run", "--host=0.0.0.0"]

