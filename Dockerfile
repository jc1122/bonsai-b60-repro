# Same public base, GPU packages and Level Zero loader used by the measured stack.
FROM vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f AS loader
FROM intel/oneapi-basekit@sha256:f62000b9049c57fc4022d3694360bc8a5c34ffee1e4726b042b1ed701d561d61
COPY --from=loader /usr/lib/x86_64-linux-gnu/libze_loader.so.1.32.0 /usr/lib/x86_64-linux-gnu/libze_loader.so.1.32.0
COPY --from=loader /usr/lib/x86_64-linux-gnu/libze_tracing_layer.so.1.32.0 /usr/lib/x86_64-linux-gnu/libze_tracing_layer.so.1.32.0
COPY --from=loader /usr/lib/x86_64-linux-gnu/libze_validation_layer.so.1.32.0 /usr/lib/x86_64-linux-gnu/libze_validation_layer.so.1.32.0
COPY runtime/debs/*.deb /tmp/neo-debs/
RUN dpkg -i /tmp/neo-debs/*.deb && \
    ln -sf libze_loader.so.1.32.0 /usr/lib/x86_64-linux-gnu/libze_loader.so.1 && \
    ln -sf libze_loader.so.1.32.0 /usr/lib/x86_64-linux-gnu/libze_loader.so && \
    ln -sf libze_tracing_layer.so.1.32.0 /usr/lib/x86_64-linux-gnu/libze_tracing_layer.so.1 && \
    ln -sf libze_validation_layer.so.1.32.0 /usr/lib/x86_64-linux-gnu/libze_validation_layer.so.1 && \
    ldconfig
LABEL org.opencontainers.image.description="Bonsai B60 reproduction compiler and runtime"
WORKDIR /repro
