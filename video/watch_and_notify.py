#!/usr/bin/env python3

import os
import sys
from inotify.adapters import Inotify
from kafka import KafkaProducer
from shutil import copyfile
import socket
import datetime
import logging
import kafka.errors as Errors
import traceback


# SETUP LOGGER
logger = logging.getLogger(__name__)
fmt_str = "[%(filename)s:line %(lineno)d] %(levelname)s:: %(message)s"
formatter = logging.Formatter(fmt_str)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.info('Video: Watch & Notify')

topic="video_curation_sched"
clientid=socket.gethostname()

kkhost=os.environ["KKHOST"]
ingestion=os.environ["INGESTION"]
in_source=os.environ["IN_SOURCE"]
video_store_dir="/var/www/mp4"

# logger.info("[VIDEO_CLIP_MSG LOG],timestamp,mode,filename")

def send_producer_msg(producer, topic, strvalue):
    try:
        ingest, filename = strvalue.split(",")
        future = producer.send(topic=topic, value=str.encode(strvalue))
        producer.flush()

        record_metadata  = future.get(timeout=10)
        # logger.info("[VIDEO_CLIP_MSG LOG],{},{},{}".format( datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ingest, filename))
        # logger.debug("[KAFKA PRODUCER MSG] Message to Ingest Pool sent")
        # logger.debug("[KAFKA PRODUCER TOPIC] {}".format(record_metadata.topic))
        # logger.debug("[KAFKA PRODUCER PARTITION] {}".format(record_metadata.partition))
        # logger.debug("[KAFKA PRODUCER OFFSET] {}\n\n".format(record_metadata.offset))
    except Errors.KafkaTimeoutError as kte:
        logger.exception("[KAFKA PRODUCER TIMEOUT ERROR]: %s", kte)
        logger.error(traceback.format_exc())
    except Errors.KafkaError as ke:
        logger.exception("[KAFKA PRODUCER ERROR]: %s", ke)
        logger.error(traceback.format_exc())
    except Exception as e:
        logger.exception("[EXCEPTION]: %s", e)
        logger.error(traceback.format_exc())

def  main(watch_folder=os.getcwd()):
    producer = KafkaProducer(bootstrap_servers=kkhost,
                client_id=clientid, api_version=(0,10))
    # logger.debug("[KAFKA PRODUCER (Video)] bootstrap_servers: {}\tclient_id: {}".format(kkhost, clientid))

    if "videos" in in_source:
        for filename in os.listdir(video_store_dir):
            if filename.endswith(".mp4"):
                for ingest in ingestion.split(','):
                    strvalue = "{},{}".format(ingest, filename)
                    send_producer_msg(producer, topic, strvalue)

                with open('/var/www/mp4/last_video.log', 'w') as f:
                    f.write("{},{},extract metadata".format(ingest, filename))

    if "stream" in in_source:
        i = Inotify()
        i.add_watch(watch_folder)

        for event in i.event_gen(yield_nones=False):
            (_, type_names, path, filename) = event
            target_dir = os.path.join(video_store_dir, filename)

            # on file write completion, we publish to topic
            if 'IN_CLOSE_WRITE' in type_names and not os.path.exists(target_dir):
                for ingest in ingestion.split(','):
                    strvalue = "{},{}".format(ingest, filename)
                    send_producer_msg(producer, topic, strvalue)
                copyfile(os.path.join(path, filename), target_dir)
                with open('/var/www/mp4/last_video.log', 'w') as f:
                    f.write("{},{},extract metadata".format(ingest, filename))

if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main()