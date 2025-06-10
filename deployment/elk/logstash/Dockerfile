
FROM docker.elastic.co/logstash/logstash:8.17.0

RUN rm -rf /usr/share/logstash/pipeline/* \
    && rm -f /usr/share/logstash/config/logstash.yml

COPY logstash.yml /usr/share/logstash/config/logstash.yml

COPY logstash-nginx.conf /usr/share/logstash/pipeline/logstash.conf

