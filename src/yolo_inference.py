from ultralytics import YOLO


# Instianlise the model
model = YOLO('yolov8x ')

# Predict model
result = model.predict('images/ollie.png', save=True)
print(result)
print('Boxes:')

for box in result[0].boxes:
    print(box)