FROM deepspeed/deepspeed:latest_torch111

ADD requirements.txt .
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r requirements.txt
RUN conda install -c bioconda blast=2.13.0
