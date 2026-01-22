FROM intellabs/vdms:v2.12.0@sha256:7b8a655ac3d42e5e86daa012762bd4d7f6e6a9faf842b0db7857fd39ea9eb45f

COPY defines.h  /vdms/src/
COPY VDMSThreadPool.h /vdms/src/
COPY VideoCommand.cc /vdms/src/
COPY VideoCommand.h /vdms/src/

COPY Video.cc /vdms/src/vcl/
COPY Video.h /vdms/include/vcl/

COPY api_schema.json /vdms/utils/src/api_schema/

RUN sed -i 's|#define DEFAULT_PATH_TMP "/tmp"|#define DEFAULT_PATH_TMP "/mnt/tmp"|g' /vdms/src/VDMSConfig.cc

RUN cd /vdms/build && rm -rf * && cmake .. && make -j && \
    echo '#!/bin/bash' > /start.sh && echo 'cd /vdms/build' >> /start.sh && \
    echo 'python3 /vdms/override_default_config.py -i /vdms/config-vdms.json -o /vdms/build/config-vdms.json' >> /start.sh && \
    echo './vdms 2>&1' >> /start.sh && chmod 755 /start.sh
CMD ["/start.sh"]
