library(keras)
library(caret)


# Não FUNCIONOU
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

# Criar um dataframe para ser usado no treinamento
train_data <- data.frame(Label = as.factor(train_labels), matrix(as.vector(train_images), nrow = nrow(train_images)))

# Criar uma função para construir o modelo
build_model <- function(units) {
  modelo <- keras_model_sequential()
  modelo %>%
    layer_conv_2d(filters = 32, 
                  kernel_size = c(3, 3), 
                  activation = 'relu', 
                  input_shape = c(28, 28, 1)) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 64, 
                  kernel_size = c(3, 3), 
                  activation = 'relu') %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_flatten() %>%
    layer_dense(units = units, activation = 'relu') %>%
    layer_dense(units = 10, activation = 'softmax')
  
  modelo %>% compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = c('accuracy')
  )
  
  return(modelo)
}

# Definir a função para treinar o modelo
train_keras <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  units <- param$units
  model <- build_model(units)
  history <- model %>% fit(x, y, epochs = 5, batch_size = 64, validation_split = 0.2, verbose = 0)
  return(list(model = model, history = history))
}

# Definir a função para prever com o modelo treinado
predict_keras <- function(model, x) {
  predict(model$model, x)
}

# Configurar o treinamento usando a função caret::train() para validação cruzada
set.seed(123)
cv <- trainControl(method = "none")  # Desativar a validação cruzada do caret

# Realizar o treinamento
resultados_treinamento <- train(
  x = train_data[, -1],  # Variáveis independentes
  y = train_data$Label,  # Variável dependente
  method = "mlpKerasDropoutCost",
  trControl = cv,
  tuneGrid = expand.grid(units = 16:64),  # Testar diferentes valores para o número de unidades
  metric = "Accuracy",
  allowParallel = TRUE,
  verbose = FALSE,
  model = "train_keras",
  predict = "predict_keras"
)

# Exibir os resultados do treinamento
print(resultados_treinamento)
