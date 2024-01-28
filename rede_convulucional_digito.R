library(tensorflow)
library(keras)
k_clear_session()
library(tidyverse)

# https://rpubs.com/sedzinfo/keras_mnist
# link acima tem a explicação

mnist <- dataset_mnist()
tri <- train_images <- mnist$train$x
train_labels <- mnist$train$y
tei <- test_images <- mnist$test$x
test_labels <- mnist$test$y

#  View(tei[1,,])
# plot(as.raster(tei[1, , ], max = 100))
#  plot(as.raster(tei[2,,],max=255))
#  plot(as.raster(tei[3,,],max=255))
#  plot(as.raster(tei[4,,],max=255))
#  plot(as.raster(tei[5,,],max=255))
#  test_labels[1]

train_images <- array_reshape(train_images, c(60000, 28 * 28))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(10000, 28 * 28))
test_images <- test_images / 255
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

network <- keras_model_sequential() %>%
    layer_dense(units = 1000, activation = "relu", input_shape = c(28 * 28)) %>%
    layer_dense(units = 1000, activation = "relu") %>%
    layer_dense(units = 10, activation = "softmax")
network %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
)
network %>% fit(train_images, train_labels, epochs = 10, batch_size = 784, verbose = 0)
network %>% evaluate(test_images,test_labels)