import json
from kafka import KafkaConsumer

class ShelfConsumer:
    def __init__(self, bootstrap_servers='localhost:9092', topic='shelf_removals', group_id='shelf_group'):
        # Kafka consumer setup
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=[bootstrap_servers],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id=group_id,
            auto_offset_reset='latest',  # Start from new messages
            enable_auto_commit=True
        )
        print("Consumer started. Listening for removals...")

    def display_removal(self, message):
        data = message.value
        print(f"[REAL-TIME ALERT] Product '{data['class_name']}' (Track ID: {data['track_id']}) removed at {data['timestamp']:.2f}")

    def run(self):
        try:
            for message in self.consumer:
                self.display_removal(message)
        except KeyboardInterrupt:
            print("Consumer stopped.")
        finally:
            self.consumer.close()

if __name__ == "__main__":
    consumer = ShelfConsumer()
    consumer.run()