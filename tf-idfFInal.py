import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer

NOME_ARQUIVO = "data_200K.csv"
COLUNA_TEXTO = 'summary' 
COLUNA_NOME_JOGO = 'name' 
COLUNA_SUMMARY = 'tags' 
TERMOS_DE_BUSCA = "dark gothic vampire world"
IDIOMA_STOPWORDS = 'english' 
NUM_PRINCIPAIS_ITENS = 10 
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

stop_words = set(stopwords.words(IDIOMA_STOPWORDS))

def clean_text(text):
    """Realiza a limpeza básica: minúsculas, remove pontuação e stopwords."""
    if pd.isna(text) or text is None:
        return ""
    

    text = str(text).lower()
  
    text = re.sub(r'[^a-z\s]', '', text) 
    
    word_tokens = word_tokenize(text)
    
    filtered_words = [w for w in word_tokens if w not in stop_words and len(w) > 1] 
    
    return " ".join(filtered_words)



try:
    df = pd.read_csv(NOME_ARQUIVO, sep=';')
    print(f"Arquivo '{NOME_ARQUIVO}' carregado com sucesso.")
except FileNotFoundError:
    print(f"Erro: O arquivo '{NOME_ARQUIVO}' não foi encontrado.")
    exit()


colunas_necessarias = [COLUNA_TEXTO, COLUNA_NOME_JOGO, COLUNA_SUMMARY]
for col in colunas_necessarias:
    if col not in df.columns:
        print(f"Erro: Coluna obrigatória '{col}' não encontrada. Colunas disponíveis: {list(df.columns)}")
        exit()


df['cleaned_tags'] = df[COLUNA_TEXTO].apply(clean_text)




vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_df=0.85, 
    min_df=5 
)


tfidf_matrix = vectorizer.fit_transform(df['cleaned_tags'])
print(f"Matriz TF-IDF criada com {tfidf_matrix.shape[0]} itens e {tfidf_matrix.shape[1]} termos/features.")


query_cleaned = clean_text(TERMOS_DE_BUSCA)
query_vector = vectorizer.transform([query_cleaned])

similarity_scores = tfidf_matrix.dot(query_vector.transpose()).toarray().flatten()

df['goth_vampire_score'] = similarity_scores
df_sorted = df.sort_values(by='goth_vampire_score', ascending=False)

print("\n" + "="*80)
print(f"TOP {NUM_PRINCIPAIS_ITENS} JOGOS MAIS RELEVANTES PARA O TEMA: '{TERMOS_DE_BUSCA}' (Baseado nas Tags)")
print("="*80)

for i in range(min(NUM_PRINCIPAIS_ITENS, len(df_sorted))):
    score = df_sorted.iloc[i]['goth_vampire_score']
    game_name = df_sorted.iloc[i][COLUNA_NOME_JOGO]
    summary = df_sorted.iloc[i][COLUNA_SUMMARY]
    original_tags = df_sorted.iloc[i][COLUNA_TEXTO]
    
    summary_display = str(summary) if pd.notna(summary) else "N/A"
    
    print(f"\n✨ RANK {i+1} | SCORE: {score:.4f} ✨")
    print("-" * 30)
    print(f"**NOME DO JOGO:** {game_name}")

    print(f"**TAGS:** {summary_display[:150]}...")
    
    print(f"**RESUMO:** {original_tags}") 
    print("-" * 30)
    
print("\n" + "="*80)