import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

raw_actions = []

def get_root_verb_lemma(text):
    doc = nlp(text.lower())
    for token in doc:
        if token.pos_ == "VERB":
            return token.lemma_
    for token in doc:
        if token.pos_ == "NOUN":
            return token.lemma_
    return text

def score_candidate(text):
    score = 0
    words = text.split()
    
    if len(words) == 1:
        score += 100
    
    if text.endswith("ing"):
        score += 50
    elif text.endswith("ed"):
        score -= 10
        
    score -= len(text)
    
    return score

def clean_strict(action_list):
    groups = {}
    
    for action in action_list:
        lemma = get_root_verb_lemma(action)
        if lemma not in groups:
            groups[lemma] = []
        groups[lemma].append(action)

    survivors = set()
    
    for lemma, candidates in groups.items():
        best_candidate = max(candidates, key=score_candidate)
        survivors.add(best_candidate)
    
    survivor_list = sorted(list(survivors))
    
    embeddings = embedder.encode(survivor_list, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings, embeddings)
    
    final_kept = set(range(len(survivor_list)))
    threshold = 0.85 
    
    for i in range(len(survivor_list)):
        if i not in final_kept: continue
        
        for j in range(i + 1, len(survivor_list)):
            if j not in final_kept: continue
            
            if cosine_scores[i][j] > threshold:
                score_i = score_candidate(survivor_list[i])
                score_j = score_candidate(survivor_list[j])
                
                if score_i >= score_j:
                    final_kept.remove(j)
                else:
                    final_kept.remove(i)
                    break
    
    final_result = sorted([survivor_list[i] for i in final_kept])

    return final_result

if __name__ == "__main__":
    cleaned = clean_strict(raw_actions)