#!/usr/bin/env python3

from collections import namedtuple
import logging
import re
import subprocess
from threading import Thread
from queue import Queue
from typing import Optional

logger = logging.getLogger(__name__)

MQTT_HOST = "citrine.home"


MqttMessage = namedtuple("MqttMessage", ["topic", "body"])

CRITERIA = {
    r'INF \[(?P<monitor>\w+): \d{1,10} - Gone into (?P<state>\w+) state': {},
    r'INF \[(?P<monitor>\w+): \d{1,10} - Closing event': {"state": "idle"}
}


def publish(message: MqttMessage):
    args = ["mosquitto_pub", "-h", MQTT_HOST, "-t", message.topic, "-m", message.body]
    process = subprocess.run(args)
    if process.returncode:
        logger.error(f"Message not published: {message.topic} - {message.body}")


def get_message_from_line(line: str) -> Optional[MqttMessage]:
    for regex, predefined_data in CRITERIA.items():
        match = re.search(regex, line)
        if not match:
            continue
        data = match.groupdict()
        data.update(predefined_data)
        return MqttMessage(topic=f"fluorite/{data['monitor']}/state", body=data["state"])


def tail_log(log_file_name, queue: Queue):
    args = ["tail", "-F", log_file_name]
    with subprocess.Popen(args, stdout=subprocess.PIPE) as process:
        while True:
            line = process.stdout.readline().decode(encoding="utf-8")
            queue.put(line)


def main():
    queue = Queue()
    t = Thread(target=tail_log, args=("/var/log/messages", queue))
    t.daemon = True
    t.start()

    while True:
        line = queue.get()
        message = get_message_from_line(line)
        if message:
            publish(message)


if __name__ == "__main__":
    main()
