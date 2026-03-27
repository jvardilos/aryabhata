FROM python

COPY train.py /
COPY requirements.txt /
RUN pip install -r requirements.txt

CMD ["python", "train.py"]
