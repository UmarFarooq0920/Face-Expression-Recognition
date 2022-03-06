import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import tensorflow.keras.layers as tfl

img_width = 96
img_height = 96

batch_size = 64

train_dir = "/content/train"
test_dir = "/content/test"

train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed=123,
    image_size=(img_height,img_width),
    batch_size=batch_size,
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
global_layer = tfl.GlobalAveragePooling2D()

IMG_SIZE = (96,96)
model = get_model(IMG_SIZE)
model.summary()

base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=1e-6)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='/content/Checkpoint/{epoch:02d}-{val_loss:.2f}-{accuracy:.2f}.hdf5',
    monitor='val_loss',
    mode='min',
    save_best_only=True)

history = model.fit(
    train_ds, epochs=20, callbacks=[model_checkpoint_callback, reduce_lr], validation_data=val_ds
)

# Fine tunning

initial_epochs = 20

base_model = model.layers[3]
base_model.trainable = True

print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

loss_function='sparse_categorical_crossentropy'

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1*base_learning_rate)

metrics=['accuracy']

model.compile(loss=loss_function,
              optimizer = optimizer,
              metrics=metrics)

fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_ds,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=val_ds,
                         callbacks=[model_checkpoint_callback, reduce_lr])

model.evaluate(test_ds)
