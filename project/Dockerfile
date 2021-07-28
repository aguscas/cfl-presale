FROM danski3456/pentaho:latest

RUN apt-get update && apt-get install -y unixodbc-dev

WORKDIR /opt/project
RUN mkdir -p savvy-freight-eta
# COPY savvy-freight-eta/setup. .
COPY setup.py .
RUN python3 -m pip install -e .
