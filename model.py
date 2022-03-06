def get_model(image_shape=IMG_SIZE):
    
    
    input_shape = IMG_SIZE + (3,)
    
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape) 

    x = preprocess_input(inputs)
    
    x = base_model(x,training=False) 
    
    x = global_layer(x)
    x = tfl.Dropout(0.2)(x)

    outputs = tfl.Dense(7,activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model