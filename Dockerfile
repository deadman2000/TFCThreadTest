FROM gcc

WORKDIR /build

RUN wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.14.0.tar.gz && \
    tar -C /usr/local -xzf *.tar.gz && \
    ldconfig && \
    rm -rf *.tar.gz

COPY model/ ./model

COPY main.cpp ./

RUN g++ main.cpp -ltensorflow -o main

CMD ["/build/main"]
