#!/usr/bin/env python3
import pika, uuid, json

class MT_RPC_Client(object):
    def __init__(self, broker, out_queue):
        self.out_queue = out_queue # where to publish requests

        # connect to broker, establish private callback_queue
        params = pika.ConnectionParameters(host=broker)
        self.connection = pika.BlockingConnection(params)
        self.channel = self.connection.channel()
        Q = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = Q.method.queue

        # listen on callback queue for responses;
        # requests are posted by __call__()
        self.channel.basic_consume\
            (queue = self.callback_queue,
             on_message_callback = self.on_response,
             auto_ack=True)
        pass

    def on_response(self, channel, method, properties, body):
        if self.correlation_id == properties.correlation_id:
            self.response = body
            pass
        return

    def __call__(self, payload):
        self.response = None
        self.correlation_id = str(uuid.uuid4())
        props = pika.BasicProperties(reply_to = self.callback_queue,
                                     correlation_id=self.correlation_id)
        self.channel.basic_publish(exchange='',routing_key=self.out_queue,
                                   properties=props,body=json.dumps(payload))
        while self.response is None:
            self.connection.process_data_events()
        return json.loads(self.response)
                                   
translate = MT_RPC_Client("localhost","MT-tasks")

test = 5 * ["Da steht ein Elephant im Kühlschrank.",
            "Im Kirschbaum saß ein Büffel, und versuchte darüber nachzudenken,"
            "wie er da wohl hingekommen sein könnte.",
            "Da muß ein Elefant im Kühlschrank sein - es sind Fußstapfen in der Butter.",
            "Ein Eichhörnchen gab sich Mühe, ihn zu trösten.",
            "Das Eichhörnchen wollte ihn so gerne trösten!"]

print("\n".join(translate(test)))
