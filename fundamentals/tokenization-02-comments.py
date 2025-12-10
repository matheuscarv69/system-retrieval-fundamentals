import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Machine learning é um campo da inteligência artificial que permite que computadores aprendam padrões a partir de dados.",
    "O aprendizado de máquina dá aos sistemas a capacidade de melhorar seu desempenho sem serem explicitamente programados.",
    "Em vez de seguir apenas regras fixas, o machine learning descobre relações escondidas nos dados.",
    "Esse campo combina estatística, algoritmos e poder computacional para extrair conhecimento.",
    "O objetivo é criar modelos capazes de generalizar além dos exemplos vistos no treinamento.",
    "Aplicações de machine learning vão desde recomendações de filmes até diagnósticos médicos.",
    "Os algoritmos de aprendizado de máquina transformam dados brutos em previsões úteis.",
    "Diferente de um software tradicional, o ML adapta-se conforme novos dados chegam.",
    "O aprendizado pode ser supervisionado, não supervisionado ou por reforço, dependendo do tipo de problema.",
    "Na prática, machine learning é o motor que impulsiona muitos avanços em visão computacional e processamento de linguagem natural.",
    "Mais do que encontrar padrões, o machine learning ajuda a tomar decisões baseadas em evidências.",
]


def preprocess(text):
    # Passo importante para limpeza do texto e padronizacao para lowercase
    # Casa != casa | essas duas palavras seriam consideradas diferentes, assim seriam dois tokens distintos
    text_lower = text.lower()

    # tokenizando o texto, cada palavra vira um token individual
    tokens = nltk.word_tokenize(text_lower)

    # Aplicando um filtro via list comprehension para manter apenas tokens alfanuméricos
    # (sem pontuação ou caracteres especiais)
    # List comprehension exemplo: [expressao for item in lista if condicao]
    return [word for word in tokens if word.isalnum()]


# Passo 1: Pre processar os documentos
# Aplicando list comprehension para preprocessar cada doc na lista documents, limpando e
# normalizando o texto e retornando uma lista de tokens alfanuméricos
# o " ".join()" serve para juntar os tokens de volta em uma string única, separando por espaço
preprocessed_docs = [" ".join(preprocess(doc)) for doc in documents]
preprocessed_docs


# Comentario referente à fit_transform e transform abaixo:
#
# O fit_transform() nos documentos, aprende o vocabulário completo e calcula os valores IDF.
# No final gera as 106 dimensões do espaço vetorial.
#
# O transform() na query, apenas transforma usando o vocabulário já aprendido.
# No final vai garantir que a query tenha as mesmas 106 dimensões dos documentos.
#
# Se você usar fit_transform() na query, vai criar um vocabulário novo só com as palavras dela.
# Isso vai impossibilitar a comparação com os documentos.
#
# O transform() vai garantir que query e documentos estejam no mesmo espaço vetorial para calcular a cosine similarity.


# Passo 2: vetorizar os documentos usando TF-IDF
# Criando o vetor TF-IDF a partir dos documentos pré-processados
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)

# Aqui temos algumas informações, mas a mais importante é a shape (11, 106)
# "Compressed Sparse Row sparse matrix of dtype 'float64' with 139 stored elements and shape (11, 106)"
# Isso significa que temos 11 documentos (linhas) e 106 termos/dimensões únicos (colunas) na matriz TF-IDF
tfidf_matrix

# Passo 3: Comparar a query com os documentos usando cosine similarity
# A ideia é encontrar quais documentos são mais similares(mais próximos) à query fornecida
query = "machine learning"
# Usamos o vectorizer para transformar a query em um vetor TF-IDF com 106 dimensões por causa do transform(), explicado acima
query_vector = vectorizer.transform([query])

# Compressed Sparse Row sparse matrix of dtype 'float64'with 2 stored elements and shape (1, 106)
# 1 linha (query) e 106 colunas (dimensões) por causa do fit_transform() nos documentos, explicado acima
query_vector

# Agora que temos a matriz TF-IDF dos documentos e o vetor TF-IDF da query,
# Podemos consultar o indice ou "medir distância" entre a query e cada documento usando cosine similarity
# Essa função retorna uma matriz 2D com as similaridades entre a query e cada documento,
# usamos o flatten() para transformar em uma lista 1D (facilitar a nossa visualização)

# array([0.23955966, 0.        , 0.26077531, 0.        , 0.        ,
#       0.2789623 , 0.        , 0.        , 0.        , 0.24103535,
#       0.26837968])

# O ponto principal é que o cosine_similarity irá medir/ajudar
# a medir as distâncias, o coseno entre a query e as dimensões dos documentos
cosine_similarity(tfidf_matrix, query_vector).flatten()
