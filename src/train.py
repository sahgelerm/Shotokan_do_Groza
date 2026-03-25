from src.data.load_data import load_images
from src.models.cnn_model import build_model
from sklearn.model_selection import train_test_split

data_dir = "data/"

X, y, classes = load_images(data_dir)

X = X / 255.0

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model = build_model(input_shape=X.shape[1:], num_classes=len(classes))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

model.save("model.keras")
