import logging
import os.path
from threading import Lock
from threading import Thread

import pika
from tensorflow.keras.models import load_model


class ActorReceiverController:
    def __init__(self, init: bool, broker_host: str, actor_model_path: str, routing_key: str, ):
        self.__broker_host = broker_host
        self.__actor_model_path = actor_model_path
        self.__routing_key = routing_key
        self.__lock = Lock()
        self.__save_lock = Lock()
        self.__latest_actor = None
        self.stop_recv = False
        self.recv_thread = None
        self.__start(init)

    def set_latest_actor(self, latest_actor):
        with self.__lock:
            self.__latest_actor = latest_actor

    def get_latest_actor(self):
        with self.__lock:
            return self.__latest_actor

    def __start(self, init: bool):
        if init:
            # Block until the controller receives the latest actor model
            self.__setup_latest_actor()
        else:
            # An actor model already exists from previous computation,
            # load it and start a new thread to subscribe for model updates
            self.set_latest_actor(load_model(self.__actor_model_path))
            self.__update_latest_actor()

    def __update_latest_actor(self):
        """
        Gets the new actor models using a receiver started in a new thread
        """
        self.__setup_exchange_and_queue(self.__update_actor_callback)
        self.recv_thread = Thread(target=self.channel.start_consuming)
        self.recv_thread.start()

    def __setup_latest_actor(self):
        """
        Blocking call to setup the current actor model
        """
        self.__setup_exchange_and_queue(self.__get_actor_and_exit_callback)
        logging.info("Waiting first actor")
        self.channel.start_consuming()

    def __setup_exchange_and_queue(self, callback):
        # Establish a connection to RabbitMQ server
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(self.__broker_host)
        )
        self.channel = self.connection.channel()

        # Declare a topic exchange named 'topic_exchange'
        self.channel.exchange_declare(exchange="topic_exchange", exchange_type="topic")

        # Declare a queue with a generated name (exclusive=True makes it exclusive to this connection)
        result = self.channel.queue_declare("", exclusive=True)
        queue_name = result.method.queue

        # Bind the queue to the topic exchange with the specified routing key
        self.channel.queue_bind(
            exchange="topic_exchange", queue=queue_name, routing_key=self.__routing_key
        )
        # Set up the consumer to use the callback function
        self.channel.basic_consume(
            queue=queue_name, on_message_callback=callback, auto_ack=True
        )

    def __get_actor_and_exit_callback(self, a, b, c, body):
        self.__save_actor(body)
        self.channel.stop_consuming()
        self.channel.close()
        self.connection.close()

    def __save_actor(self, body):
        with self.__save_lock:
            with open(self.__actor_model_path, "wb") as actor_model_file:
                actor_model_file.write(body)
            actor_model_file.close()

    def __update_actor_callback(self, a, b, c, body):
        if not self.stop_recv:
            self.__save_actor(body)
            self.set_latest_actor(load_model(self.__actor_model_path))
            logging.info("Actor updated")
        else:
            self.channel.stop_consuming()
            self.channel.close()
            self.connection.close()
