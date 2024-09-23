def wordTableBuild(all_sentences, max_table_cap):
    word2frequency = {}
    beginWordMap = {}
    for sentence in all_sentences:
        for c in sentence:
            if c == ' ':
                continue
            if c not in word2frequency:
                word2frequency[c] = 1
                beginWordMap[c] = [c]
            else:
                word2frequency[c] += 1
    while len(word2frequency.keys()) < max_table_cap:
        new_word2frequency = {}
        for sentence in all_sentences:
            for idx, c in enumerate(sentence):
                if c == ' ':
                    continue
                #ss = sentence.substr(idx)
                ss = sentence[idx:]
                for word in beginWordMap.get(c, []):
                    if ss.startswith(word):
                        next_c = ss[len(word)]
                        for next_word in beginWordMap.get(next_c, []):
                            if ss.startswith(word + next_word):
                                if (word + next_word) not in new_word2frequency:
                                    new_word2frequency[word + next_word] = 1
                                else:
                                    new_word2frequency[word + next_word] += 1
        
        sorted(new_word2frequency)
        new_word = list(new_word2frequency.keys())[-1]
        word2frequency[new_word] = 1
        beginWordMap[new_word[0]].append(new_word)
    
    return list(word2frequency.keys())

if __name__ == '__main__':
    all_sentences = ["cats sit on the mat", "dogs sit on the mat", "cats like the cat"]
    words = wordTableBuild(all_sentences, 100)
    print(words)