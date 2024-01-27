#!/usr/bin/env python
import pika

# connection = pika.BlockingConnection(
#     pika.ConnectionParameters(host='localhost'))
# channel = connection.channel()
#
# channel.exchange_declare(exchange='logs', exchange_type='fanout')
#
# message = ' '.join(sys.argv[1:]) or "info: Hello World!"
# channel.basic_publish(exchange='logs', routing_key='', body=message)
# print(f" [x] Sent {message}")
# connection.close()

connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()

# Declare a topic exchange named 'topic_exchange'
channel.exchange_declare(exchange="topic_exchange", exchange_type="topic")

# Specify the routing key (topic) for the message
routing_key = "actor-model"

# Message to be published
message_body = "Hello, RabbitMQ!"

# Publish the message to the topic exchange with the specified routing key
channel.basic_publish(
    exchange="topic_exchange", routing_key=routing_key, body=message_body
)

print(f" [x] Sent '{message_body}' with routing key '{routing_key}'")

# Close the connection
connection.close()
