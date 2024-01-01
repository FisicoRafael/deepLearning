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
    para_camadas <- c("neuronios_por_camada", "ativacao_por_camada", "dropout")
    if (!is.list(config_camadas) ||
        !all(para_camadas %in% names(config_camadas))) {
        stop("Os parâmetros de configuração das CAMADAS devem ser fornecidos como uma lista contendo 'neuronios_por_camada', 'ativacao_por_camada' e 'dropout'.")
    }

    para_compile <- c("num_entradas", "num_saidas", "otimizador", "funcao_loss", "metrica", "kernel_inicializador")
    if (!is.list(config_compile) ||
        !all(para_compile %in% names(config_compile))) {
        stop("Os parâmetros de configuração do COMPILE devem ser fornecidos como uma lista contendo 'num_entradas', 'num_saidas', 'otimizador','funcao_loss','metrica' e 'kernel_inicializador'.")
    }

    modelo <- keras_model_sequential()

    for (i in 1:length(config_camadas$neuronios_por_camada)) {
        if (i == 1) {
            modelo %>%
                layer_dense(
                    units = config_camadas$neuronios_por_camada[i], input_shape = c(config_compile$num_entradas),
                    activation = config_camadas$ativacao_por_camada[i],
                    kernel_initializer = config_compile$kernel_inicializador
                )
        } else {
            if (!is.null(config_camadas$dropout) && !is.na(config_camadas$dropout[i - 1])) {
                # print(config_camadas$dropout[i - 1])
                modelo %>%
                    layer_dense(
                        units = config_camadas$neuronios_por_camada[i], activation = config_camadas$ativacao_por_camada[i]
                    ) %>%
                    layer_dropout(rate = config_camadas$dropout[i - 1])
            } else {
                modelo %>%
                    layer_dense(
                        units = config_camadas$neuronios_por_camada[i], activation = config_camadas$ativacao_por_camada[i]
                    )
            }
        }
    }

    modelo <- modelo %>%
        compile(
            loss = config_compile$funcao_loss, # Função de perda para classificação multiclasse
            optimizer = config_compile$optimizer, # Otimizador, por exemplo, Adam
            metrics = config_compile$metrics # Métrica a ser avaliada durante o treinamento
        )

    return(modelo)
}

# config_camadas_2 <- list(
#     neuronios_por_camada = c(2, 3, 1),
#     ativacao_por_camada = c("relu", "relu", "sigmoid"),
#     dropout = c()
# )

# config_compile_2 <- list(
#     num_entradas = 10,
#     num_saidas = 1,
#     otimizador = "adam",
#     funcao_loss = "binary_crossentropy",
#     metrica = list("accuracy"),
#     kernel_inicializador = "glorot_uniform"
# )


# build_RNA(
#     config_camadas = config_camadas_2,
#     config_compile = config_compile_2
# )
