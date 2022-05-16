import pymorphy2
import re
from nltk.corpus import stopwords
import torch
import requests
from googlesearch import search
from bs4 import BeautifulSoup
from rusenttokenize import ru_sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
# from laserembeddings import Laser
from torch.utils.data import DataLoader
from dataset_reader import MyDataset
from BERT_model import predict, device, Classifier

morph = pymorphy2.MorphAnalyzer()
tokensre = re.compile(r'\b[а-яА-ЯёЁйЙ]+\b')


def get_tokens(text):
    return [morph.parse(word)[0].normal_form for word in
            list(filter(lambda word: word not in stopwords.words('russian'),
                        tokensre.findall(text)))]


def text_preprocess(text):
    text = text.replace('\n', ' ').replace('\xa0', ' ').replace('[en]', ' ').replace('(', '').replace(')', '')
    text = re.sub(r'\[\d+\]', ' ', text)
    return ' '.join(text.split())


def tanimoto_func(s1, s2):
    a, b, c = len(s1), len(s2), 0.0

    for sym in s1:
        if sym in s2:
            c += 1

    return c / (a + b - c)


def mean_pooling(model_output, attention_mask, doc_class):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    sentences_embeddings = sum_embeddings / sum_mask
    if doc_class:
        return torch.sum(sentences_embeddings, 0) / sentences_embeddings.size()[0]
    else:
        return sentences_embeddings.double()


def get_embed(model, tokenizer, sentences: list, max_length=512, doc_class=False):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    text_embedding = mean_pooling(model_output, encoded_input['attention_mask'], doc_class)
    return text_embedding


def get_texts(request):
    pages = []
    search_results = search(request.strip())
    cnt = 0
    for url in search_results:
        if cnt < 3:
            if url.startswith('http') and '.ua' not in url:
                try:
                    respond = requests.get(url)
                except:
                    continue
                soup = BeautifulSoup(respond.text)
                l = soup.find_all('p')
                pages.append(l)
                cnt += 1
            else:
                continue

    texts = ''
    for page in pages:
        page_text = [text_preprocess(block.text) for block in page]
        texts += ' '.join(page_text)
    return texts


def get_class(request):
    texts = get_texts(request)

    splitted_texts = ru_sent_tokenize(texts.replace(' .', '.'))
    splitted_texts.append(request)
    processed_text = list(filter(lambda x: x, [' '.join(get_tokens(sentence)) for sentence in splitted_texts]))

    #  1 TF-IDF
    vectorizer = TfidfVectorizer()
    TFIDF_matrix = vectorizer.fit_transform(processed_text)
    request_tf_idf_vector = TFIDF_matrix[-1].toarray()

    tf_idf_cos_distances = []
    for tf_idf_vector in TFIDF_matrix[:-1]:
        tf_idf_cos_distances.append(1 - distance.cosine(tf_idf_vector.toarray(), request_tf_idf_vector))

    #  2 Tanimoto coefficient
    tanimoto_distances = []
    request_tanimoto_elems = get_tokens(processed_text[-1])
    for sentence in processed_text[:-1]:
        tanimoto_distances.append(tanimoto_func(get_tokens(sentence), request_tanimoto_elems))

    #  3 Laser embeddings (BiLSTM)
    # laser = Laser()
    #
    # embeddings = laser.embed_sentences(splitted_texts, lang='ru')
    #
    # request_embedding = embeddings[-1]
    # cos_distance_embeddigs = [1 - distance.cosine(embed, request_embedding) for embed in embeddings[:-1]]

    #  4 BERT embeddings
    model_path = 'models/rubert-base-cased-sentence'
    tokenizer = AutoTokenizer.from_pretrained(model_path, force_download=False)
    model = AutoModel.from_pretrained(model_path, force_download=False)

    bert_embeddings = get_embed(model, tokenizer, splitted_texts)

    request_bert_embedding = bert_embeddings[-1].tolist()
    cos_distance_bert_embeddigs = [1 - distance.cosine(embed.tolist(), request_bert_embedding) for embed in
                                   bert_embeddings]

    #  5 BM25
    processed_query = processed_text[-1]
    flag = False
    try:
        bm25 = BM25Okapi([get_tokens(sentence) for sentence in processed_text[:-1]])
    except:
        flag = True
    if flag:
        bm25_distance = [0] * len(cos_distance_bert_embeddigs)
    else:
        bm25_distance = bm25.get_scores(get_tokens(processed_query))

    #  Взвешанная оценка
    def get_method_norm(vec):
        max_elem = max(vec)
        min_elem = min(vec)
        vec_norm = [(elem - min_elem) / (max_elem - min_elem) if max_elem - min_elem != 0 else 0 for elem in vec]

        return vec_norm

    sn1 = get_method_norm(tf_idf_cos_distances)
    sn2 = get_method_norm(tanimoto_distances)
    # sn3 = get_method_norm(cos_distance_embeddigs)
    sn4 = get_method_norm(cos_distance_bert_embeddigs)
    if not flag:
        sn5 = get_method_norm(bm25_distance)
    else:
        sn5 = [0] * len(sn1)

    score_final = [1 / 5 * sn1[idx] + 1 / 5 * sn2[idx]  + 1 / 5 * sn4[idx] + 1 / 5 * sn5[idx] for idx
                   in range(len(processed_text[:-1]))]

    score_final.sort(reverse=True)
    max_relevant = splitted_texts[score_final.index(score_final[0])] + ' ' + \
                   splitted_texts[score_final.index(score_final[1])] + ' ' + \
                   splitted_texts[score_final.index(score_final[2])]

    tf_idf_cos_distances.sort(reverse=True)
    tf_idf = splitted_texts[tf_idf_cos_distances.index(tf_idf_cos_distances[0])] + ' ' + \
             splitted_texts[tf_idf_cos_distances.index(tf_idf_cos_distances[1])] + ' ' + \
             splitted_texts[tf_idf_cos_distances.index(tf_idf_cos_distances[2])]

    tanimoto_distances.sort(reverse=True)
    tanimoto = splitted_texts[tanimoto_distances.index(tanimoto_distances[0])] + ' ' + \
               splitted_texts[tanimoto_distances.index(tanimoto_distances[1])] + ' ' + \
               splitted_texts[tanimoto_distances.index(tanimoto_distances[2])]

    # cos_distance_embeddigs.sort(reverse=True)
    # embeddigs = splitted_texts[cos_distance_embeddigs.index(cos_distance_embeddigs[0])] + ' ' + \
    #             splitted_texts[cos_distance_embeddigs.index(cos_distance_embeddigs[1])] + ' ' + \
    #             splitted_texts[cos_distance_embeddigs.index(cos_distance_embeddigs[2])]

    cos_distance_bert_embeddigs.sort(reverse=True)
    bert_embeddigs = splitted_texts[cos_distance_bert_embeddigs.index(cos_distance_bert_embeddigs[0])] + ' ' + \
                     splitted_texts[cos_distance_bert_embeddigs.index(cos_distance_bert_embeddigs[1])] + ' ' + \
                     splitted_texts[cos_distance_bert_embeddigs.index(cos_distance_bert_embeddigs[2])]

    bm25_distance = list(bm25_distance)
    bm25_distance.sort(reverse=True)
    bm25 = splitted_texts[bm25_distance.index(bm25_distance[0])] + ' ' + \
           splitted_texts[bm25_distance.index(bm25_distance[1])] + ' ' + \
           splitted_texts[bm25_distance.index(bm25_distance[2])]

    model = torch.load('models/pretrained_model/model.pth', map_location=torch.device(device))
    texts = (
        max_relevant, tf_idf, tanimoto, bert_embeddigs, bm25
    )

    dict_scores = {0: 0, 1: 0}
    for text in texts:
        test_data = [(text, request)]
        test_ds = MyDataset(range(len(test_data)), test_data, [0])
        test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)
        pred = predict(model, test_dl)
        dict_scores[0] = max(dict_scores[0], pred[2][0][0])
        dict_scores[1] = max(dict_scores[1], pred[2][0][1])

    return 0 if dict_scores[0] > dict_scores[1] else 1
