import json
import unicodedata
import string
from difflib import SequenceMatcher
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

def remover_acentos(s: str) -> str:
    s = unicodedata.normalize('NFD', s)
    return ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')

def limpar_tokens(txt: str):
    txt = txt.lower()
    txt = remover_acentos(txt)
    txt = ''.join(c for c in txt if c not in string.punctuation)
    return txt.split()

def limpar_texto(txt: str):
    txt = txt.lower()
    txt = remover_acentos(txt)
    return txt

def overlap_coef(tokens_a, tokens_b):
    a = set(tokens_a)
    b = set(tokens_b)
    if not a or not b:
        return 0.0
    return len(a.intersection(b)) / min(len(a), len(b))

def jaccard(tokens_a, tokens_b):
    a = set(tokens_a)
    b = set(tokens_b)
    if not a and not b:
        return 1.0
    union = a.union(b)
    if not union:
        return 0.0
    return len(a.intersection(b)) / len(union)

def seq_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()

# carrega FAQ
with open('faq.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
PERGUNTAS = data['perguntas']
RESPOSTAS = data['respostas']

faq_texts = [limpar_texto(t) for _, t in sorted(PERGUNTAS.items(), key=lambda x: int(x[0]))]
vectorizer = TfidfVectorizer().fit(faq_texts)
faq_vectors = vectorizer.transform(faq_texts)

@app.route('/api/query', methods=['POST'])
def query():
    payload = request.get_json(force=True)
    q = payload.get('q', '')
    if not q:
        return jsonify({'error': 'query vazia'}), 400

    q_tokens = limpar_tokens(q)
    q_text_clean = limpar_texto(q)
    q_vec = vectorizer.transform([q_text_clean])

    scores = {}
    for idx_str, pergunta_text in PERGUNTAS.items():
        idx = int(idx_str) - 1
        p_tokens = limpar_tokens(pergunta_text)
        p_text_clean = limpar_texto(pergunta_text)

        js = jaccard(q_tokens, p_tokens)
        sm = seq_ratio(q_text_clean, p_text_clean)
        ol = overlap_coef(q_tokens, p_tokens)
        sem = float(cosine_similarity(q_vec, faq_vectors[idx])[0,0])

        combined = 0.45 * js + 0.25 * sm + 0.10 * ol + 0.20 * sem
        scores[idx_str] = combined

    best_idx = max(scores, key=scores.get)
    best_score = scores[best_idx]
    best_answer = RESPOSTAS[best_idx]

    if best_score > 0.60:
        resposta = best_answer
    elif best_score < 0.25:
        resposta = 'Essa pergunta vai além da minha capacidade.'
    else:
        resposta = 'Especifique melhor sua pergunta, por favor.'

    return jsonify({
        'answer': resposta,
        'confidence': best_score
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
