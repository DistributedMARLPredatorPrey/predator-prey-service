import logging
import sys
import socket
import requests

from src.main.model.config.config_utils import PredatorPreyConfig


def check_pika_consumer(
        rabbitmq_host="localhost",
        rabbitmq_port=15672,
        rabbitmq_user="guest",
        rabbitmq_password="guest",
):
    # Use RabbitMQ Management API for getting a list of consumers
    api_url = f"http://{rabbitmq_host}:{rabbitmq_port}/api/consumers"
    response = requests.get(api_url, auth=(rabbitmq_user, rabbitmq_password))
    # To verify the local actor receiver is subscribed and ready,
    # check if some consumer has the local IP address.
    if response.status_code == 200:
        json_arr = response.json()
        if len(json_arr) > 0:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            if any([local_ip in json["channel_details"]["connection_name"] for json in json_arr]):
                sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    learner_conf = PredatorPreyConfig().learner_service_configuration()
    check_pika_consumer(rabbitmq_host=learner_conf.pubsub_broker)
