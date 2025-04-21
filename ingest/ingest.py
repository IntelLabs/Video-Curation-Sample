#!/usr/bin/python3

from kafka import KafkaConsumer
from subprocess import call
from zkstate import ZKState
import traceback
import socket
import time
import os
import logging
import kafka.errors as Errors

num_cpus=int(os.environ["NCPU"])
if num_cpus > 0:
    from multiprocessing import cpu_count
    import random
    import psutil
    cpu_nums = list(range(psutil.cpu_count()))
    random.shuffle(cpu_nums)
    proc = psutil.Process(os.getpid())
    proc.cpu_affinity(cpu_nums[:num_cpus])


# SETUP LOGGER
logger = logging.getLogger(__name__)
fmt_str = "[%(filename)s:line %(lineno)d] %(levelname)s:: %(message)s"
formatter = logging.Formatter(fmt_str)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.info('INGEST POOL')

topic="video_curation_sched"
groupid="curaters"
clientid=socket.gethostname()

producer_topic="ingest_sched"

kkhost=os.environ["KKHOST"]
vdhost=os.environ["VDHOST"]
dbhost=os.environ["DBHOST"]
in_source=os.environ["IN_SOURCE"]

def send_producer_msg(producer, topic, strvalue):
    try:
        producer.send(topic=topic, value=str.encode(strvalue))
        producer.flush()
    except Errors.KafkaTimeoutError as kte:
        logger.exception("[KAFKA PRODUCER TIMEOUT ERROR]: %s", kte)
        logger.error(traceback.format_exc())
    except Errors.KafkaError as ke:
        logger.exception("[KAFKA PRODUCER ERROR]: %s", ke)
        logger.error(traceback.format_exc())
    except Exception as e:
        logger.exception("[EXCEPTION]: %s", e)
        logger.error(traceback.format_exc())

while True:
    try:
        c=KafkaConsumer(topic,bootstrap_servers=kkhost,
               client_id=clientid, group_id=groupid, auto_offset_reset="earliest",
               api_version=(0,10))

        for msg in c:
            mode,clip_name=msg.value.decode('utf-8').split(",")
            zk=ZKState("/state/"+clip_name,mode)
            if not zk.processed():
                if zk.process_start():

                    logger.info("Processing {}: {}...".format(clip_name, mode))
                    
                    while True:
                        logger.info("Downloading "+clip_name)
                        sts=call(["/usr/bin/wget","-O",clip_name,vdhost+"/mp4/"+clip_name])
                        if sts==0: break
                        time.sleep(0.5)

                    call(["/opt/gstreamer_gva/metaData_extract","-i",clip_name,"-n","-x",mode,"-a",dbhost,"-l"])
                    
                    os.remove(clip_name)
                    zk.process_end()
            zk.close()

    except:
        logger.error(traceback.format_exc())
