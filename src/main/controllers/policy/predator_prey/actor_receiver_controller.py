import os.path
from threading import Thread

import pika
from tensorflow.keras.models import load_model
from threading import Lock


class ActorReceiverController:
    def __init__(self, broker_host: str, actor_model_path: str, routing_key: str):
        self.broker_host = broker_host
        self.actor_model_path = actor_model_path
        self.routing_key = routing_key
        self.__lock = Lock()
        self.__latest_actor = None
        if os.path.exists(actor_model_path):
            self.set_latest_actor(load_model(actor_model_path))
            self.__setup_receiver()
        else:
            self.__setup_latest_actor()

    def set_latest_actor(self, latest_actor):
        with self.__lock:
            self.__latest_actor = latest_actor

    def get_latest_actor(self):
        with self.__lock:
            return self.__latest_actor

    def __setup_receiver(self):
        self.__update_latest_actor()
        Thread(target=self.__consume).start()

    def __update_latest_actor(self):
        """
        Gets the new actor models using a receiver started in a new thread
        """
        self.__setup_exchange_and_queue(self.__update_actor)

    def __setup_latest_actor(self):
        """
        Blocking call to setup the current actor model
        """
        self.__setup_exchange_and_queue(self.__get_latest_actor_and_exit)
        print("waiting first actor")
        self.__consume()

    def __setup_exchange_and_queue(self, callback):
        # Establish a connection to RabbitMQ server
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(self.broker_host)
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

    def __get_latest_actor_and_exit(self, ch, method, properties, body):
        with open(self.actor_model_path, "wb") as actor_model_file:
            actor_model_file.write(body)
        print("Actor received")
        self.connection.close()

    def __update_actor(self, ch, method, properties, body):
        with open(self.actor_model_path, "wb") as actor_model_file:
            actor_model_file.write(body)
        self.set_latest_actor(load_model(self.actor_model_path))
        print("Actor updated")

    def __consume(self):
        # Start consuming messages
        self.channel.start_consuming()
