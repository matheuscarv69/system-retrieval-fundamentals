import os
import nltk
import shutil
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser

import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

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
    tokens = [word for word in tokens if word.isalnum()]

    # Removendo stopwords em português, mas mantendo "e", "ou", "não"
    stopwords = set(nltk.corpus.stopwords.words("portuguese")) - {"e", "ou", "não"}
    tokens = [word for word in tokens if word not in stopwords]
    return tokens


text = "Machine learning é um campo da inteligência artificial. que permite que computadores aprendam padrões a partir de dados."
preprocess(text)

# Este bloco verifica se já existe um diretório chamado "index_dir".
# Se existir, ele remove o diretório e todo o seu conteúdo usando `shutil.rmtree("index_dir")`.
# Em seguida, cria um novo diretório chamado `index_dir` com `os.mkdir("index_dir")`.
if os.path.exists("index_dir"):
    shutil.rmtree("index_dir")
os.mkdir("index_dir")

# Cria um Schema com dois campos: "title" é um ID (armazenado e único),
# isto é, indexa o valor inteiro como um token (usando IDAnalyzer) e o armazena;
# "content" é um campo de texto (TEXT) que também é armazenado e usa o
# analisador/formatos padrões para indexação de texto.
schema = Schema(title=ID(stored=True, unique=True), content=TEXT(stored=True))

# Cria um índice no diretório "index_dir" usando o schema definido acima.
index = create_in("index_dir", schema)

# Obtém um writer para o índice e percorre `documents` com um índice `i`.
writer = index.writer()
for i, doc in enumerate(documents):
    # Para cada `doc` adiciona um documento ao índice com dois campos:
    # - title: a string do número `i` (usado aqui como identificador/rotulo)
    # - content: o texto do documento
    writer.add_document(title=str(i), content=doc)
# Finalmente, `writer.commit()` persiste as mudanças e libera o índice.
writer.commit()

query = "machine E learning"


# Função para realizar busca booleana no índice
def boolean_search(query, index):
    # Cria um QueryParser para interpretar a string de busca.
    # - "content" é o campo do schema que será consultado.
    # - schema=index.schema fornece ao parser informações sobre analisadores e tipos de campo.
    parser = QueryParser("content", schema=index.schema)

    # Converte a string `query` em um objeto de consulta (parsed_query) que o searcher pode executar.
    # O parser trata tokenização, normalização e operadores booleanos (AND/OR/NOT), conforme definido pelo schema.
    parsed_query = parser.parse(query)

    # Abre um searcher (objeto responsável por executar buscas) no índice.
    # O uso de "with" garante que o searcher seja fechado automaticamente ao sair do bloco,
    # mesmo se ocorrerem erros — isso evita vazamento de recursos ou locks no índice.
    with index.searcher() as searcher:
        # Executa a busca usando a consulta já parseada; retorna um objeto Results contendo os hits ordenados por relevância.
        results = searcher.search(parsed_query)
        # Cada 'hit' no 'results' representa um documento que corresponde à consulta.
        # Um 'hit' funciona parecido com um dicionário/objeto: permite acessar os campos armazenados
        # (por exemplo "title" e "content") e metadados como 'score' (relevância).
        # Constrói e retorna uma lista de tuplas (title, content) a partir dos hits.
        return [(hit["title"], hit["content"]) for hit in results]


boolean_search(query, index)
