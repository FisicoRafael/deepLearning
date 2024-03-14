# Load the libraries
library(tensorflow)
library(keras)
k_clear_session()
library(tidyverse)

library(rpart)
library(rpart.plot)
library(caret)


#Load the dataframe
base <- read.csv("data/wdbc.csv")

base <- base %>%
    mutate(
        Diagnosis = case_when(
            Diagnosis == "B" ~ 1,
            TRUE ~ 0
        ),
        Diagnosis = as.numeric(Diagnosis)
    )

# Create training dataframe and test dataframe
porcentagem_treinamento <- 0.75

base_treinamento <- base %>%
    sample_frac(porcentagem_treinamento)

base_teste <- base %>%
    anti_join(base_treinamento, by = "id")

base_treinamento$id <- NULL
base_teste$id <- NULL

#
base_treinamento_feature <- base_treinamento %>%
    select(-all_of("Diagnosis"))
base_treinamento_outcome <- base_treinamento %>%
    select(all_of("Diagnosis"))
base_teste_feature <- base_teste %>%
    select(-all_of("Diagnosis"))
base_teste_outcome <- base_teste %>%
    select(all_of("Diagnosis"))

#
num_input <- ncol(base_treinamento_feature)
num_output <- ncol(base_treinamento_outcome)
num_layer <- round((num_input + num_output) / 2)

#Building the neural network
model <- keras_model_sequential() %>%
    layer_dense(
        units = num_layer, activation = "relu",
        kernel_initializer = "random_uniform",
        input_shape = num_input
    ) %>%
    layer_dense(
        units = num_layer, activation = "relu",
        kernel_initializer = "random_uniform"
    ) %>%
    layer_dense(
        units = num_output, activation = "sigmoid"
    )

custom_optimizer <- optimizer_adam(
    learning_rate = 0.001,
    weight_decay = 0.0001,
    clipvalue = 0.5
)

model %>% compile(
    loss = "binary_crossentropy", # Loss Function for Multiclass Classification
    optimizer = custom_optimizer, # Optimizer, e.g. Adam
    metrics = list("binary_accuracy") # Metric to be evaluated during training
)

# Carry out training
model %>% fit(
    as.matrix(base_treinamento_feature),
    as.matrix(base_treinamento_outcome),
    epochs = 20, # Número de épocas de treinamento
    batch_size = 10 # Tamanho do batch
)

evaluate <- model %>%
    evaluate(as.matrix(base_teste_feature),
        as.matrix(base_teste_outcome),
        batch_size = 10
    )

previsao <- model %>%
    predict(as.matrix(base_teste_feature),
        batch_size = 100
    ) %>%
    as_tibble()

previsao$V1 <- previsao$V1 <= 0.5
base_teste_outcome$Diagnosis <- base_teste_outcome$Diagnosis == 0


matriz_confusao <- table(
    base_teste_outcome$Diagnosis,
    previsao$V1
)
cf <- confusionMatrix(matriz_confusao)

# library(ggplot2)

# # Criar um gráfico de barras para representar a matriz de confusão
# ggplot(
#     data = as.data.frame(cf$table),
#     aes(x = value_real, y = predicao, fill = Freq)
# ) +
#     geom_tile() +
#     geom_text(aes(label = Freq)) +
#     labs(
#         title = "Matriz de Confusão",
#         x = "Real",
#         y = "Predito"
#     ) +
#     scale_fill_gradient(low = "lightblue", high = "darkblue") +
#     theme_minimal()
