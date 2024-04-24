import sys

import requests

from src.main.model.config.config_utils import ConfigUtils


def check_pika_consumer(
    queue_name,
    rabbitmq_host="localhost",
    rabbitmq_port=15672,
    rabbitmq_user="guest",
    rabbitmq_password="guest",
):
    # RabbitMQ Management API endpoint for getting a list of consumers
    api_url = f"http://{rabbitmq_host}:{rabbitmq_port}/api/consumers"

    # Make a request to the RabbitMQ Management API using basic authentication
    response = requests.get(api_url, auth=(rabbitmq_user, rabbitmq_password))

    # Check if the request was successful (HTTP status code 200)
    if response.status_code == 200:
        json_arr = response.json()
        if len(json_arr) > 0:
            sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    learner_conf = ConfigUtils().learner_service_configuration()

    # # Establish a connection to RabbitMQ server
    # connection = pika.BlockingConnection(
    #     pika.ConnectionParameters(learner_conf.pubsub_broker)
    # )
    # channel = connection.channel()
    # # Declare a topic exchange named 'topic_exchange'
    # channel.exchange_declare(exchange="topic_exchange", exchange_type="topic")
    #
    # # Declare a queue with a generated name (exclusive=True makes it exclusive to this connection)
    # result = channel.queue_declare("", exclusive=True)
    # queue_name = result.method.queue
    #
    # # # Bind the queue to the topic exchange with the specified routing key
    # # channel.queue_bind(
    # #     exchange="topic_exchange", queue=queue_name, routing_key=""
    # # )
    #
    # if len(channel.consumer_tags) == 0:
    #     LOGGER.info("Nobody is listening.  I'll come back in a couple of minutes.")
    #     # Failure
    #     sys.exit(1)
    # sys.exit(0)

    # Example usage
    queue_name_to_check = ""
    check_pika_consumer(queue_name_to_check, rabbitmq_host=learner_conf.pubsub_broker)
