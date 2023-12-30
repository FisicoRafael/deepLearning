source("export_libs.R")
source("build_rna/build_rna.R")

backend <- "tensorflow"
k_set_image_data_format("channels_last")
Sys.setenv(KERAS_BACKEND = backend)


base <- read.csv("data/wdbc.csv")

base <- base %>%
    mutate(
        Diagnosis = case_when(
            Diagnosis == "B" ~ 1,
            TRUE ~ 0
        ),
        Diagnosis = as.numeric(Diagnosis)
    ) %>%
    select(-all_of("id"))

base_feature <- base %>%
    select(-all_of("Diagnosis"))
base_outcome <- base %>%
    select(all_of("Diagnosis"))

num_input <- ncol(base_feature)
num_output <- ncol(base_outcome)
num_layer <- round((num_input + num_output) / 2)

a <- build_RNA()

# constriuondo agora a validação cruzada
# Defina os índices para a validação cruzada
set.seed(123)
folds <- createFolds(base_outcome$Diagnosis, k = 10) # 5-fold cross-validation

# Inicialize vetores para armazenar resultados
accuracy_values <- numeric(length(folds))

# Iteração sobre os folds para treinar e avaliar o modelo
for (i in seq_along(folds)) {
    # Separa os dados em treino e teste para esta fold
    train_indices <- unlist(folds[-i])
    test_indices <- folds[[i]]

    x_train <- as.matrix(base_feature[train_indices, ])
    y_train <- as.matrix(base_outcome[train_indices, ])
    x_test <- as.matrix(base_feature[test_indices, ])
    y_test <- as.matrix(base_outcome[test_indices, ])

    # Constrói o modelo
    model <- build_RNA()

    # Treina o modelo
    model %>% fit(
        x_train, y_train,
        epochs = 10,
        batch_size = 32,
        verbose = 0
    )

    # Avalia o modelo
    scores <- model %>% evaluate(x_test, y_test, verbose = 0)
    accuracy_values[i] <- scores[[2]]
}

mean_accuracy <- mean(accuracy_values)
print(paste("Accuracy:", mean_accuracy))


soma()