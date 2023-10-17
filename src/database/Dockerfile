FROM postgres:latest

RUN apt-get update && apt-get install -y postgresql-server-dev-all make gcc git

RUN git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
RUN cd pgvector && make && make install
