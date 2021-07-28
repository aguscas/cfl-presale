FROM microsoft/mssql-server-linux:2017-latest 

COPY setup.sql setup.sql
COPY setup_database.sh setup_database.sh
COPY entrypoint.sh entrypoint.sh

CMD /bin/bash ./entrypoint.sh
