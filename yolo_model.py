from keras import backend as k 
from keras.applications import VGG16

image_size = 608, 608, 3
label_size = 19, 19, 55
learning_rate = 0.0001
num_epochs = 1500
minibatch_size = 32
print_cost = True
X_train = None
Y_train = None
X_test = None
Y_test = None

def  yolo_model (train_shape):
    img = input(train_shape)
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=image_size,
                      name = 'vgg_base_net')(img)
    #freeze all the layers in the vgg16 network
    for layer in conv_base.layers:
        layer.trainable = False
    x = k.Conv2D(128, (1,1), strides = (1,1), padding = 'same', name = 'conv1')(conv_base)
    x = k.Conv2D(55, (1,1), strides = (1,1), padding = 'same', name = 'conv2')(x)
    model = k.Model(inputs = img, outputs = x, name='yolo')
    return model

##########################################################
# compute cost function to be written 
def yolo_cost (Y_hat, Y):
    cost = None
    return cost
##########################################################
##create the model
yolo = yolo_model(X_train.shape[1:])
#determine adam optimizer for our model
optimizer = k.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#comiple the model
yolo.compile(loss=yolo_cost, optimizer=optimizer)
#train the model
yolo.fit(X_train, Y_train, epochs=num_epochs, batch_size=minibatch_size)
