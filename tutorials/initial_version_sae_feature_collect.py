eature_intersections = []
stds = []

assert sae

with torch.no_grad(): 
    player_count = 0
    for name, char in players.items():
        player_count += 1
        print(f'Processing player {player_count} of {NUM_PLAYERS}')
        trait = char['trait']



        ##EXTRACTING VALUE DATA
        value_count = 0
        for value_name, group in grouped:
            if value_count > 5:
                break
            value_count += 1
            print(value_name)

            groupagreementall = group['agreement']
            groupquestionall = group['question']
            groupitemall = group['item']

            scores = []
            question_batch_no = math.ceil(len(groupagreementall) / MAX_QUESTIONS_PER_BATCH)
            for qbn in range(question_batch_no):
                groupagreement = groupagreementall[qbn * MAX_QUESTIONS_PER_BATCH : (qbn+1) * MAX_QUESTIONS_PER_BATCH]
                groupquestion = groupquestionall[qbn * MAX_QUESTIONS_PER_BATCH : (qbn+1) * MAX_QUESTIONS_PER_BATCH]
                groupitem = groupitemall[qbn * MAX_QUESTIONS_PER_BATCH : (qbn+1) * MAX_QUESTIONS_PER_BATCH]

                questions = []
                answers = []
                for groupmember in zip(groupagreement, groupquestion, groupitem):
                    a = groupmember[0]
                    q = groupmember[1]
                    qi = groupmember[2]

                    if SAMPLE_MODE == 'PROMPT_SAE_INTERV':
                        prompt, role_setting = generate_question_norole(a, q, qi, ALLOW_UNSURE_ANSWER)
                    elif SAMPLE_MODE == 'ROLES':
                        prompt, role_setting = generate_question(name, trait, a, q, qi, ALLOW_UNSURE_ANSWER)    
                    else:
                        raise ValueError('Invalid SAMPLE_MODE')
                    questions.append(prompt)
                    answers.append(a)
        ##EXTRACTING VALUE DATA END

                gen_answers = []                    
                
                if SAMPLE_MODE == 'ROLES':
                    #while True:
                    if sae:
                        questions_tokens = model.tokenizer(questions, return_tensors="pt", padding=True, truncation=True)
                        gen_tokens = model.generate(questions_tokens.input_ids, max_new_tokens=20, verbose=False)
                        # gen_tokens_f = model.forward(questions_tokens.input_ids, return_type='logits', attention_mask=questions_tokens.attention_mask)
                        # gen_tokens_hf = hf_model.generate(questions_tokens.input_ids, attention_mask=questions_tokens.attention_mask, max_new_tokens=10)
                        gen_texts = model.tokenizer.batch_decode(gen_tokens)
                        #gen_texts_f = model.tokenizer.batch_decode(gen_tokens_f[:,-1,:].max(dim=-1).indices)
                        questions_padded = model.tokenizer.batch_decode(questions_tokens.input_ids)
                        for question, gen_text in zip(questions_padded, gen_texts):
                            gen_answer = judge_answer(gen_text.lower()[len(question):].strip())
                            gen_answers.append(gen_answer)
                    else:
                        assert False
                        gen_texts = model(questions, max_new_tokens=5)
                        gen_answers = [gen_text[0]["generated_text"][len(question):].lower() for question, gen_text in zip(questions, gen_texts)]

                    #print(gen_answers)
                    #if all([judge_positive(gen_answer) or judge_negative(gen_answer) or (ALLOW_UNSURE_ANSWER and gen_answer.startswith('unsure')) for gen_answer in gen_answers]):
                    #if all([judge_positive(gen_answer) or judge_negative(gen_answer) or judge_unsure(gen_answer) for gen_answer in gen_answers]):
                    #    break
                    
                else:
                    raise ValueError('Invalid SAMPLE_MODE')
                
                for ga, answer in zip(gen_answers, answers):
                    if ga == 'yes':
                        scores.append(answer)
                    elif ga == 'no':
                        scores.append(-answer)
                    elif ga == 'unsure':
                        scores.append(0)
                    else:
                        raise ValueError('Invalid answer')
            assert len(scores) == len(groupagreementall)

            if SCORE_GRANULARITY == 'question':
                assert False
                for q, s in zip(groupquestionall, scores):
                    players[name][q] = s
            elif SCORE_GRANULARITY == 'value':
                if SAMPLE_MODE == 'ROLES':
                    players[name][value_name] = sum(scores) / len(scores)
                #print(value_name, scores, 'std: ', np.std(scores))
                stds.append(np.std(scores))
            else:
                raise ValueError('Invalid SCORE_GRANULARITY')
        
        steer_dim_results.append(steer_dim_result)

        def get_cache_feature(prompt, feature_ids):
            results = {}
            for sf in feature_ids:
                results[sf] = 0
            logits, cache = model.run_with_cache(prompt, prepend_bos=True)
            layer_cache = cache[sae.cfg.hook_name]
            feature_acts = sae.encode(layer_cache)
            for sf in feature_ids:
                for token in feature_acts[0]:
                    results[sf] += token[sf].item()
            return results
        
        def get_cached_feature_of_value(prompt, token_id):
            logits, cache = model.run_with_cache(prompt, prepend_bos=True)
            layer_cache = cache[sae.cfg.hook_name]
            feature_acts = sae.encode(layer_cache)
            sae_out = sae.decode(feature_acts)

            index_of_value = model.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist().index(token_id)
            feature_of_value = feature_acts[0][index_of_value]
            topk_values, topk_indice = torch.topk(feature_of_value, int(0.005 * len(feature_of_value)))
            
            zero_index = torch.where(topk_indice == 0)
            if len(zero_index[0]) == 0:
                ti = topk_indice.tolist()
            else:
                ti = topk_indice[:zero_index[0]].tolist()

            del cache
            #flatten the dimension 1 and 2 of feature_acts
            #feature_acts = feature_acts.flatten(1, 2)
            #select the 0.5% highest-activated elements in feature_acts[0][0]
            #topk_values, topk_indice = torch.topk(feature_acts, int(0.005 * len(feature_acts[0][0])))
            #filter out the indices with 0 value
            return feature_of_value, ti

        if sae:
            pass
            # if SAE_FEATURE_SOURCE == 'COLLECT':    
            #     if base_model == 'GPT2-SMALL':
            #         token_id = 3815
            #     elif base_model == 'GEMMA-2B-IT':
            #         token_id = 1618 #1261
            #     elif base_model == 'GEMMA-2B':
            #         token_id = 4035
            #     else:f
            #         raise ValueError('Invalid base model')
                
            #     feature_of_value, ti = get_cached_feature_of_value(questions_padded[-1], token_id)
            #     for ttii in ti:
            #         players[name][ttii] = feature_of_value[ttii].item()
            #     feature_intersections.append(set(ti))
            # elif SAE_FEATURE_SOURCE == 'FIX':
            #     results = get_cache_feature(questions_padded[-1], FIXED_SAE_FEATURES)
            #     for sf in FIXED_SAE_FEATURES:
            #         players[name][sf] = results[sf]
            # else:
            #     raise ValueError('Invalid SAE_FEATURE_SOURCE')
            
    if sae and SAMPLE_MODE== 'ROLES':
        feature_intersection = set.intersection(*feature_intersections)
        feature_union = set.union(*feature_intersections)
    print('stds: ', np.mean(stds))

    

=========================   ==========================   ==========================
with torch.no_grad():
    # activation store can give us tokens.

    #value_of_interest = 'Calmness'
    #value_of_interest = 'Causality:Interactionism'
    #value_of_interest_ref = 'Authority'
    # feature_of_interest = 15509 #3376

    #value_of_interest = 'Financial Prosperity'
    value_of_interest = 'Decisiveness'
    feature_of_interest = 10096

    df_valuebench = pd.read_csv(os.path.join(LOCAL_SAE_MODEL_PATH, 'value_data/value_orientation.csv'))
    grouped = df_valuebench.groupby('value')
    name = 'Nicholas Davis'
    trait = "Gender: male; Job: Financial risk analyst; DOB: 1942-03-14; bio: Nicholas Davis is a male financial risk analyst with a medium level of responsibility. Known for his calm and collected demeanor, Nicholas approaches his work with a low level of aggression, always seeking to find strategic solutions to potential financial risks. With a keen eye for detail and strong analytical skills, he is dedicated to ensuring that his clients make informed decisions to protect their investments."

    for value_name, group in grouped:
        if value_name != value_of_interest:
            continue
        
        for trial in range(10):
            questions = []
            answers = []
            randomno = random.randint(0, len(group['agreement'])-1)
            groupagreement = group['agreement'][randomno:randomno+1]
            groupquestion = group['question'][randomno:randomno+1]
            groupitem = group['item'][randomno:randomno+1]

            for groupmember in zip(groupagreement, groupquestion, groupitem):
                a = groupmember[0]
                q = groupmember[1]
                qi = groupmember[2]

                question = generate_question(name, trait, a, q, qi, allow_unsure)
                questions.append(question)
                answer = a
                answers.append(answer)

            assert sae
            if base_model == 'GPT2-SMALL':
                token_id = 3815
            elif base_model == 'GEMMA-2B-IT':
                token_id = 1618 #1261
            elif base_model == 'GEMMA-2B':
                token_id = 4035
            elif base_model == 'MISTRAL-7B':
                token_id = 3815##TBD
            else:
                raise ValueError('Invalid base model')

            batch_tokens = model.tokenizer(questions, return_tensors="pt", padding=True).input_ids#[0].tolist().index(3815)
            index_of_value = [tks.index(token_id) for tks in batch_tokens.tolist()][-1]
            #logits, cache = model.run_with_cache(batch_tokens, prepend_bos=True)
            logits, cache = model.run_with_cache(batch_tokens)

            # Use the SAE
            feature_acts = sae.encode(cache[sae.cfg.hook_name])
            #feature_acts[0][index_of_value][feature_of_interest] = 0#feature_acts[0][index_of_value][feature_of_interest] * 10
            sae_out = sae.decode(feature_acts)
            
            def computed_cache(batch_tokens):
                intermidiate_data = batch_tokens
                intermidiate_data = model.hook_embed(model.embed(intermidiate_data))
                
                for layer in range(13):
                    block = model.blocks[layer]
                    intermidiate_data = block(intermidiate_data)
                return intermidiate_data
            
            def computed_modified_cache(sae_out):
                intermidiate_data = sae_out
                for layer in range(13, 18):
                    block = model.blocks[layer]
                    intermidiate_data = block(intermidiate_data)
                return model.unembed(model.ln_final(intermidiate_data))
                
            # gen_tokens = model.generate(batch_tokens, max_new_tokens=5, verbose=False)
            # # gen_tokens_f = model.forward(questions_tokens.input_ids, return_type='logits', attention_mask=questions_tokens.attention_mask)
            # # gen_tokens_hf = hf_model.generate(questions_tokens.input_ids, attention_mask=questions_tokens.attention_mask, max_new_tokens=10)
            # gen_texts = model.tokenizer.batch_decode(gen_tokens)
            # gen_answer = gen_texts[-1].lower()[len(questions[-1]):].strip()
            
            pre = computed_cache(batch_tokens)
            assert torch.all(pre==cache[sae.cfg.hook_name])
            
            cmc = computed_modified_cache(sae_out)  
            #cmc = computed_modified_cache(pre)  


            modified_answer = model.tokenizer.batch_decode([torch.argmax(cmc[0][-1])])[0].strip().lower()
            original_answer = model.tokenizer.batch_decode([torch.argmax(logits[0][-1])])[0].strip().lower()
            
            print('standard_answer: ', answers[0])
            print('original answer: ', original_answer)
            print('modified answer: ', modified_answer)
            print('\n')    
            del  cache, logits, feature_acts, sae_out #pre, cmc
            gc.collect()
        print('=========================')
        '''
        # save some room
        del cache
        activations = []
        for iovn in range(len(index_of_value)):
            feature_of_value = feature_acts[iovn][index_of_value[iovn]]
            pass
            activations.append(feature_of_value[feature_of_interest].item())
            
        px.histogram(activations, nbins=100).show()
        #px histogram x label more fine grained
        
        
        # # ignore the bos token, get the number of features that activated in each token, averaged accross batch and position
        # l0 = (feature_acts[:, 1:] > 0).float().sum(-1).detach()
        # print("average l0", l0.mean().item())
        # px.histogram(l0.flatten().cpu().numpy()).show()
        '''