import pika
import json
import time
import os
from datetime import datetime

# --- CONFIGURATION ---
# In Docker, the hostname is the service name from docker-compose.yml ("rabbitmq")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
QUEUE_NAME = "audit_trail"
LOG_FILE = "medical_audit.jsonl"  # JSON Lines format (one JSON object per line)


def connect_to_rabbitmq():
    """
    Attempts to connect to RabbitMQ with a retry mechanism.
    Crucial for Docker environments where RabbitMQ might take a few seconds to boot.
    """
    retries = 5
    while retries > 0:
        try:
            print(f"Attempting to connect to RabbitMQ at {RABBITMQ_HOST}...")
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=RABBITMQ_HOST)
            )
            print("Successfully connected to RabbitMQ!")
            return connection
        except pika.exceptions.AMQPConnectionError:
            print(f"RabbitMQ not ready yet. Retrying in 5 seconds... ({retries} attempts left)")
            retries -= 1
            time.sleep(5)

    raise Exception(" CRITICAL: Could not connect to RabbitMQ after multiple attempts.")


def callback(ch, method, properties, body):
    """
    This function fires every time a new message arrives in the queue.
    """
    try:
        # 1. Decode the message
        message = json.loads(body.decode('utf-8'))

        # 2. Add a secure, server-side timestamp
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "PREDICTION_LOG",
            "payload": message
        }

        # 3. Write to the audit file (Append mode)
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")

        print(f" AUDIT SAVED: {audit_entry['timestamp']} | Diagnosis: {message.get('diagnosis', 'UNKNOWN')}")

        # 4. Tell RabbitMQ we successfully processed the message so it can delete it from the queue
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print(f"ERROR processing message: {e}")
        # Do not ack the message if it failed, so RabbitMQ requeues it
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)


def main():
    print("🛡️ Starting Medical Audit Logger Microservice...")

    # Connect and setup channel
    connection = connect_to_rabbitmq()
    channel = connection.channel()

    # Declare the queue (ensures it exists even if the Gateway hasn't created it yet)
    # durable=True ensures messages survive if RabbitMQ restarts
    channel.queue_declare(queue=QUEUE_NAME, durable=True)

    # Tell RabbitMQ to only send one message at a time to this worker
    channel.basic_qos(prefetch_count=1)

    # Start listening
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)

    print(f"🎧 Listening for events on queue '{QUEUE_NAME}'... To exit press CTRL+C")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print(" Shutting down logger...")
        channel.stop_consuming()
    finally:
        connection.close()


if __name__ == "__main__":
    main()