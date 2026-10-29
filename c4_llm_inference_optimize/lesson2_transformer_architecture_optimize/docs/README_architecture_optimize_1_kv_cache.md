# Transformer Architecture Transformations: KV CACHING

## 🎯 1) KV Cache: RELEVANT CODE
Both these codes below u can see that when KV cache is used
- Latency: improves
- Throughput: improves
- Peak GPU MEmory increases: i.e deteriorates. this is the tradeoff

Code Notebooks
- you can use kv_cache by simply setting use_cache = True in model.generate(..., use_cache= True,...). **model.generate(...) will do all the heavy lifting for us**. SO GREAT !
- using model "Qwen/Qwen2.5-0.5B": [c4_llm_inference_optimize/lesson2_transformer_architecture_optimize/ksw_solution/L2E1a_quantify_kv_cache_growth_ksw_soln.ipynb](../ksw_solution/L2E1a_quantify_kv_cache_growth_ksw_soln.ipynb)
- using model "facebook/opt-125m" : [c1_model_optimize_foundation/lesson1_intro_to_model_optimization/T1_latency_throughput/exercise1_latency_throughput_ksw_solution.ipynb](../../../c1_model_optimize_foundation/lesson1_intro_to_model_optimization/T1_latency_throughput/exercise1_latency_throughput_ksw_solution.ipynb)



## 🎯 2) KV Cache: Check your intuition
![alt text](readme_imgs/r1a_kv_cache_math/kv_cache_1.png) 

## 🎯 3) Calculating KV Cache: Formula
Also Compare Qwen vs Llam
![alt text](readme_imgs/r1a_kv_cache_math/kv_cache_2.png) 
![alt text](readme_imgs/r1a_kv_cache_math/kv_cache_3.png) 

## 🎯 4) KV Cache and Number of Tokens
How is grows Qwen vs Llama
![alt text](readme_imgs/r1a_kv_cache_math/kv_cache_4.png)

### 🎯 4.1) KV Cache : Real Tokens vs. Padding Tokens
Instead of Real Tokens, lets say you use EOS tokens as padding tokens. Does that Affect KV Cache ? \
Answer: No it doesnt affect KV Cache. But it could affect accuracy of the output
![alt text](readme_imgs/r1a_kv_cache_math/kv_cache_5.png) 
![alt text](readme_imgs/r1a_kv_cache_math/kv_cache_6.png)

## 🎯 5) HOW KV CACHING WORKS
- only Query vectors are new for every new token
- But KV are the same, for previous tokens also. Store previous KVs
- Note that Q1*K1, Q2*K1, Q2*K2 are not stored. The dot products are new foe every new Query
     - Query1 deals with Q1*K1
     - Query2 deals with Q2*K1, Q2*K2 
     - Query2 deals with Q3*K1, Q3*K2, Q3*K3

### 🎯 5.1) K, V values are stored. Q*K products are not stored, Q not stored
![alt text](readme_imgs/r1b_kv_cache_workings/kv_what_is_stored_1.png) 
![alt text](readme_imgs/r1b_kv_cache_workings/kv_what_is_stored_2.png) 
![alt text](readme_imgs/r1b_kv_cache_workings/kv_what_is_stored_3.png) 

### 🎯 5.2) Visual Representation
![alt text](readme_imgs/r1b_kv_cache_workings/kv_what_is_stored_4.png) 

### 🎯 5.3) Lets dumb it down. what are Q, K, V in terms of words
![alt text](readme_imgs/r1b_kv_cache_workings/kv_what_is_stored_5.png)

### 🎯 5.4) Reiterating: Previous Attention Scores are NOT NEEDED i.e previous Q*K not needed
![alt text](readme_imgs/r1b_kv_cache_workings/kv_what_is_stored_6.png)
![alt text](readme_imgs/r1b_kv_cache_workings/kv_what_is_stored_7.png)

### 🎯 5.5) What aspect of the previous token is needed ? WHAT EXACTLY IS NEEDED
- token 1 is needed to generate token2: 
    - But only K1, V1 of token1 is needed. Q1 is not needed, Attention score Q1*K1 is not needed. The token1 itself or its embedding is not needed.
- token2 and token1 are needed to generate token3
    - But only K1, V1 of token1 is needed. Q1 is not needed. Attention score Q1*K1 is not needed. The token1 itself or its embedding is not needed.
    - But only K2, V2 of token2 is needed. Q2 is not needed. Attention score Q2*K1, Q2*K2 is not needed. The token2 itself or its embedding is not needed.

![alt text](readme_imgs/r1b_kv_cache_workings/kv_what_is_stored_8.png)

### 🎯 5.6) Q1*K2, Q1*K3, Q2*K3 etc is NEVER NEEDED (for causal self attention)
- token1 cannot attend to token2, or to token3 or to token4: Q1*K2, Q1*K3, Q1*K4 Not needed
- token2 cannnot only attend to token1, so Q2*K1, Q2*K2 is fine. It cannot attend to token3, token4 etc   Q2*K3, Q2*K4 etc.  Not needed
![alt text](readme_imgs/r1b_kv_cache_workings/what_is_needed_causal_attention.png)

### 🎯 5.7) token1 can attend token2 in non causal attention/ cross attention etc
![alt text](readme_imgs/r1b_kv_cache_workings/what_is_needed_non_causal_attention.png)

## 🎯 6) Udacity's  KV Caching Lessons
UDACITY explains kv caching with diagrams with a lot of matrices. its essentially the same as above. do not get confused by the different steps for each token and the matrices

Udacity is trying to show
- token 1: K1,V1 is calculated.        Then Q1*K1 is calculated. then softmax (Q1*K1)*V1 is calculated
- token 2: K1,V1 is calculated **again**.  K2,V2 is calculated new. then Q2*K1, Q2*K2 etc is calculated
- token 3: K1,V1, K2, V2 are calculated **again**.  K3,V3 is calculated new. then Q3*K1, Q3*K2, Q3*K4 etc is calculated

The key is we want to avoid all the repeated calculations of K1,V1, K2,V2 etc. and cache them. Don't get confused/distracted by all the attention score matrices etc

#### 🎯 Token1 generation: without KV Cache

![alt text](readme_imgs/r1c_udacity_kv_cache/annotated_no_kv_cache_token1.jpg) 

#### 🎯 Token2 generation: without KV Cache
![alt text](readme_imgs/r1c_udacity_kv_cache/annotated_no_kv_cache_token2.jpg) 

#### 🎯 Token3 generation: without KV Cache
![alt text](readme_imgs/r1c_udacity_kv_cache/annotated_no_kv_cache_token3.jpg)

#### 🎯 Token4 generation: without KV Cache
Everything that went in generating token1 , token2, token3 and token4 is here
![alt text](readme_imgs/r1c_udacity_kv_cache/annotated_no_kv_cache_token4a.jpg) 

But specifically these are the only steps that happened after token3 , for token4
![alt text](readme_imgs/r1c_udacity_kv_cache/annotated_no_kv_cache_token4b.jpg) 

#### 🎯 Token4 generation: with KV Cache
![alt text](readme_imgs/r1c_udacity_kv_cache/annotated_with_kv_cache_token4a.jpg)
![alt text](readme_imgs/r1c_udacity_kv_cache/annotated_with_kv_cache_token4b.jpg)

## 🎯 7) KV Caching : Short Summary
source: below images are from udacity. Pasting here for my quick reference/ learning
![alt text](readme_imgs/r1c_udacity_kv_cache/kv_cache_short_summary1.png) 

You can understand the tradeof right
PROS: 
- you are saving so much time, compute, compute cost and computer energy by caching KV
- **REDUCE COMPLEXITY FROM O(N*N) TO O(N)
- CONs: Imagine storing all those KV values especialy when the context is the order of 1000s of words. You are looking at K and V values that is more than the weights of the model
![alt text](readme_imgs/r1c_udacity_kv_cache/kv_cache_short_summary2.png)


## 🎯 8) KV Caching : Long Summary
source: below images are from udacity. Pasting here for my quick reference/ learning
![alt text](readme_imgs/r1c_udacity_kv_cache/kv_cache_long_summary1.png) 
![alt text](readme_imgs/r1c_udacity_kv_cache/kv_cache_long_summary2.png)


