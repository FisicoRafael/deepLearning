library(tensorflow)
library(keras)
k_clear_session()
library(tidyverse)

# Carregar o conjunto de dados MNIST
mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

# Pré-processamento dos dados
train_images <- array_reshape(train_images, c(nrow(train_images), 28, 28, 1))
test_images <- array_reshape(test_images, c(nrow(test_images), 28, 28, 1))

# Normalizar os valores dos pixels para o intervalo [0, 1]
train_images <- train_images / 255
test_images <- test_images / 255

# Criar um modelo sequencial
modelo <- keras_model_sequential()

# Adicionar uma camada de convolução 2D ao modelo
modelo %>%
    layer_conv_2d(
        filters = 32,
        kernel_size = c(3, 3),
        activation = "relu",
        input_shape = c(28, 28, 1)
    )

modelo %>%
    layer_batch_normalization()

# Adicionar uma camada de pooling 2D para redução de dimensionalidade
modelo %>%
    layer_max_pooling_2d(pool_size = c(2, 2))

# Adicionar mais camadas de convolução e pooling, se necessário
modelo %>%
    layer_conv_2d(
        filters = 64,
        kernel_size = c(3, 3),
        activation = "relu"
    )

modelo %>%
    layer_max_pooling_2d(pool_size = c(2, 2))

# Adicionar camada de flatten para transformar o tensor 3D em vetor 1D
modelo %>%
    layer_flatten()

# Adicionar camada totalmente conectada (dense) com função de ativação 'relu'
modelo %>%
    layer_dense(units = 64, activation = "relu")

modelo %>%
    layer_dropout(rate = 0.2)

# Camada de saída com 10 unidades (um para cada classe) e função de ativação 'softmax'
modelo %>%
    layer_dense(units = 10, activation = "softmax")

# Compilar o modelo
modelo %>% compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = c("accuracy")
)

# Treinar o modelo
modelo %>% fit(train_images, train_labels, epochs = 5, batch_size = 64, validation_split = 0.2)

# Avaliar o modelo no conjunto de teste
modelo %>% evaluate(test_images, test_labels)
