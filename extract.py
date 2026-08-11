import json
import os
from collections import Counter
import re

json_files = [
    'PAB/annotation/train/attr_0.json', 'PAB/annotation/train/attr_1.json',
    'PAB/annotation/train/attr_2.json', 'PAB/annotation/train/attr_3.json',
    'PAB/annotation/train/attr_4.json', 'PAB/annotation/train/attr_5.json',
    'PAB/annotation/train/attr_6.json', 'PAB/annotation/train/attr_7.json',
]

output_json = 'action_list.json'

BLACKLIST_KEYWORDS = [
    "failure", "failed", "fails", "unable", "trying", "attempting", 
    "appears", "seems", "likely", "possibly", "intended", 
    "wrong", "unexpected", "unknown", "unnecessary",
    "too much", "too hard", "too fast", "too heavy",
    "the man", "the woman", "the baby", "the child", "the person",
    "sentence", "caption", "description"
]

NOUN_BLACKLIST = {
    "wind", "snow", "rain", "trash", "water", "soda", "salt", "shadow", 
    "wedding", "toast", "stick", "step", "slow", "split", "sour", 
    "shocked", "scared", "upset", "thinking", "starting", "stopping"
}

def clean_text(text):
    text = text.lower().strip()
    text = text.rstrip(".,;!?")
    return text

def is_valid_action(text):
    words = text.split()
    if len(words) > 3:
        return False
    
    start_blacklist = ("the ", "a ", "an ", "his ", "her ", "their ", "my ", "our ")
    if text.startswith(start_blacklist):
        return False
    
    for bad_word in BLACKLIST_KEYWORDS:
        if bad_word in text:
            return False

    if text in NOUN_BLACKLIST:
        return False

    return True

def scan_actions(files):
    action_counter = Counter()
    
    print(f"Start scanning {len(files)} files...")
    
    for file_path in files:
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except:
                    f.seek(0)
                    data = [json.loads(line) for line in f if line.strip()]
            
            if isinstance(data, dict) and 'annotations' in data:
                data = data['annotations']
            
            for item in data:
                candidates = []
                if 'anomaly' in item and item['anomaly']:
                    candidates.append(str(item['anomaly']))
                if 'normal' in item and item['normal']:
                    candidates.append(str(item['normal']))
                
                for raw_text in candidates:
                    sub_items = raw_text.split(',')
                    for sub in sub_items:
                        cleaned = clean_text(sub)
                        if cleaned and is_valid_action(cleaned):
                            action_counter[cleaned] += 1
                        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    final_actions = []
    
    MIN_FREQUENCY = 2 
    
    for action, count in action_counter.most_common():
        if count >= MIN_FREQUENCY:
            final_actions.append(action)
        else:
            pass

    final_actions = sorted(final_actions)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(final_actions, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    scan_actions(json_files)