FROM floydhub/pytorch
# gcr.io/deeplearning-platform-release/pytorch-gpu.1-1
RUN apt-get install software-properties-common
RUN add-apt-repository ppa:openjdk-r/ppa
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y openjdk-8-jdk

RUN git clone https://github.com/kermitt2/grobid \
&& cd grobid \
&& ./gradlew clean install \
&& git clone https://github.com/kermitt2/grobid-ner.git \
&& cd grobid-ner \
&& cp -r resources/models/* ../grobid-home/models/ \
&& ./gradlew clean install \
&& cd ../../

RUN git clone https://github.com/kermitt2/entity-fishing \
&& cd entity-fishing/data/db/ \
&& wget https://science-miner.s3.amazonaws.com/entity-fishing/0.0.4/linux/db-kb.zip \
&& wget https://science-miner.s3.amazonaws.com/entity-fishing/0.0.4/linux/db-en.zip \
&& wget https://science-miner.s3.amazonaws.com/entity-fishing/0.0.4/linux/db-fr.zip \
&& wget https://science-miner.s3.amazonaws.com/entity-fishing/0.0.4/linux/db-de.zip \
&& for file in *.zip; do unzip $file; rm $file; done \
&& cd ../.. \
&& ./gradlew clean build \
&& cd ..

# ARG SSH_PRIVATE_KEY
# RUN mkdir /root/.ssh/
# RUN echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa

# RUN chmod 700 /root/.ssh/id_rsa
# RUN chown -R root:root /root/.ssh

# RUN touch /root/.ssh/known_hosts
# RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN git clone https://github.com/duynht/epfl-industry-graph.git

# RUN rm /root/.ssh/id_rsa

RUN cd epfl-industry-graph
RUN pip install -r requirements.txt

CMD ../entity-fishing/gradlew appRun & bash







