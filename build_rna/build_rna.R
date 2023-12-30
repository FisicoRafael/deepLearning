
#' Título da função
#'
#' Descrição detalhada da função, seus parâmetros e retorno.
#'
#' @param parametro1 Descrição do parâmetro 1
#' @param parametro2 Descrição do parâmetro 2
#'
#' @return Descrição do que a função retorna
#'
#' @examples
#' nome_da_funcao(3, 5)
#'



build_RNA <- function(
    config_camadas,
    config_compile) {
    # validação

    # Validando os parâmetros de configuração de camadas
    if (!is.list(config_camadas) ||
        !all(names(config_camadas) %in% c("neuronios_por_camada", "ativacao_por_camada", "dropout"))) {
        stop("Os parâmetros de configuração de camadas devem ser fornecidos como uma lista contendo 'neuronios_por_camada', 'ativacao_por_camada' e 'dropout'.")
    }



    model <- keras_model_sequential()

    layer_dense(
        units = neurons, activation = activation,
        kernel_initializer = kernel_initializer,
        input_shape = num_input
    ) %>%
        layer_dropout(0.2) %>%
        layer_dense(
            units = neurons, activation = "relu",
            kernel_initializer = "random_uniform"
        ) %>%
        layer_dense(
            units = num_output, activation = "sigmoid"
        ) %>%
        compile(
            loss = "binary_crossentropy", # Função de perda para classificação multiclasse
            optimizer = optimezer, # Otimizador, por exemplo, Adam
            metrics = list("binary_accuracy") # Métrica a ser avaliada durante o treinamento
        )

    return(model)
}
