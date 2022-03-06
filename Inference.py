def prediction(image_path):
    img = tf.keras.utils.load_img(
          image_path, target_size=(img_height, img_width))

    img = tf.keras.utils.img_to_array(img)
        
    plt.title('Image')
    plt.axis('off')
    plt.imshow((img/255.0).squeeze())
        
    predict = model.predict(img[np.newaxis , ...])
    predicted_class = labels[np.argmax(predict[0] , axis = -1)]
        
    print('Prediction Value: ' , np.max(predict[0] , axis = -1))
    print("Classified:",predicted_class)