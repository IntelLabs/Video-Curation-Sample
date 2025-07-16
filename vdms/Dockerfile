FROM intellabs/vdms:v2.11.0

# COPY VideoCommand.h /vdms/src/
COPY VideoCommand.cc /vdms/src/
COPY Video.cc /vdms/src/vcl/
COPY Video.h /vdms/include/vcl/
COPY DynamicMetadataHandler.h /vdms/src/
COPY DynamicMetadataHandler.cc /vdms/src/
COPY api_schema.json /vdms/utils/src/api_schema/
COPY defines.h  /vdms/src/

RUN sed -i '/src\/VideoCommand.cc/a \    src/DynamicMetadataHandler.cc' /vdms/CMakeLists.txt

RUN cd /vdms/build && rm -rf * && cmake .. && make -j && \
    echo '#!/bin/bash' > /start.sh && echo 'cd /vdms/build' >> /start.sh && \
    echo 'python3 /vdms/override_default_config.py -i /vdms/config-vdms.json -o /vdms/build/config-vdms.json' >> /start.sh && \
    echo './vdms 2>&1' >> /start.sh && chmod 755 /start.sh
CMD ["/start.sh"]
