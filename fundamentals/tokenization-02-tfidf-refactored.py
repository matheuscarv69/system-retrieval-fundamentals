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

# Passo 2: vetorizar os documentos usando TF-IDF
# Criando o vetor TF-IDF a partir dos documentos pré-processados
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)

# Passo 3: Comparar a query com os documentos usando cosine similarity
# A ideia é encontrar quais documentos são mais similares(mais próximos) à query fornecida
query = "machine learning"


def search_tfidf(query, vectorizer, tfidf_matrix):
    query_vector = vectorizer.transform([query])

    # O ponto principal é que o cosine_similarity irá medir/ajudar
    # a medir as distâncias, o coseno entre a query e as dimensões dos documentos
    similarities = cosine_similarity(tfidf_matrix, query_vector).flatten()

    # Passo 4: Ordenar os documentos pela similaridade
    # Teremos uma lista de tuplas usando enumerate para associar o score de similaridade ao índice do documento
    sorted_similiarities = list(enumerate(similarities))

    # Imprimindo o sorted_similarities teremos essa saida:
    # [(0, np.float64(0.2395596642426218)),
    # (1, np.float64(0.0)),
    # (2, np.float64(0.26077531168763424)),
    # (3, np.float64(0.0)),
    # (4, np.float64(0.0)),
    # (5, np.float64(0.27896229878819406)),
    # (6, np.float64(0.0)),
    # (7, np.float64(0.0)),
    # (8, np.float64(0.0)),
    # (9, np.float64(0.2410353474383299)),
    # (10, np.float64(0.26837968022989944))]

    # Por padrão, o sorted irá ordernar pelo primeiro elemento da tupla (índice do documento)
    # Queremos ordenar a lista de tuplas (idx, score) pelo segundo elemento (score).
    # A lambda `lambda x: x[1]` retorna esse segundo elemento para cada tupla usando uma função anônima "x: x[1],
    # e reverse=True garante ordem decrescente (maior score primeiro).
    results = sorted(sorted_similiarities, key=lambda x: x[1], reverse=True)
    return results

search_similarities = search_tfidf(query, vectorizer, tfidf_matrix)
search_similarities

print(f"Top 10 documents by similarity score for query: '{query}'\n")

for doc_idx, score in search_similarities[:10]:
    print(f"Document: {doc_idx} -> {documents[doc_idx]}\n Similarity Score: {score}\n")