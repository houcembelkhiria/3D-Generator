from app.events.factory import get_event_bus
bus = get_event_bus()
id = bus.publish({"url": "test", "name": "test", "scene": "new", "has_texture": False})
print("Published:", id)
bus.ack(id)
events = bus.consume()
print("Pending events after ack:", len(events))
