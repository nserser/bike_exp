FROM rust:1.85

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages --no-cache-dir pandas numpy matplotlib scipy tomli pyarrow

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked

WORKDIR /work
RUN cp /build/target/release/dfrcert /usr/local/bin/dfrcert

ENTRYPOINT ["/usr/local/bin/dfrcert"]
