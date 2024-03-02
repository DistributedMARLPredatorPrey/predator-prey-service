import os
from threading import Thread

import pika
import tensorflow as tf


class ActorReceiverController:

    def __init__(self, actor_model_path: str, routing_key: str):
        self.actor_model_path = actor_model_path
        self.routing_key = routing_key
        self._setup_latest_actor()
        self.update_latest_actor()

    def update_latest_actor(self):
        """
        Gets the new actor models using a receiver started in a new thread
        """
        self._setup_exchange_and_queue(self._update_actor)
        Thread(target=self._consume()).start()

    def _setup_latest_actor(self):
        """
        Blocking call to setup the current actor model
        """
        if os.path.exists(self.actor_model_path):
            self.latest_actor = tf.keras.models.load_model(self.actor_model_path)
        else:
            # Block the thread until an actor model is received
            self._setup_exchange_and_queue(self._get_latest_actor_and_exit)
            self._consume()

    def _setup_exchange_and_queue(self, callback):
        # Establish a connection to RabbitMQ server
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters("localhost")
        )
        self.channel = self.connection.channel()

        # Declare a topic exchange named 'topic_exchange'
        self.channel.exchange_declare(exchange="topic_exchange", exchange_type="topic")

        # Declare a queue with a generated name (exclusive=True makes it exclusive to this connection)
        result = self.channel.queue_declare("", exclusive=True)
        queue_name = result.method.queue

        # Bind the queue to the topic exchange with the specified routing key
        self.channel.queue_bind(
            exchange="topic_exchange", queue=queue_name, routing_key=self.routing_key
        )
        # Set up the consumer to use the callback function
        self.channel.basic_consume(
            queue=queue_name, on_message_callback=callback, auto_ack=True
        )

    def _get_latest_actor_and_exit(self, ch, method, properties, body):
        # TODO: what's the received format?
        self.latest_actor = body
        # Save actor model in h5 file
        print(f" [x] First actor: {body}")
        self.connection.close()

    def _update_actor(self, ch, method, properties, body):
        self.latest_actor = body
        # Save actor model in h5 file
        print(f" [x] Update actor: {body}")

    def _consume(self):
        # Start consuming messages
        self.channel.start_consuming()


# rec = ActorReceiverController("/home/luca/Desktop/model.h5",
#                               "actor-model")
# print("done")