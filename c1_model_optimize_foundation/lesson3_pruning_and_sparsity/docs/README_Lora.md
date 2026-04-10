# 🎯  LORA: LOW RANK ADAPTATION

## 🎯 References/ Citations
This README is intended to (quickly) document and compile my learning on LORA from various sources. \
LORA: Info and Screenshots from the following
- EDWARD HU: https://www.youtube.com/watch?v=DhRoTONcyZE&t=34s
- Mark Hennings: https://www.youtube.com/watch?v=t1caDsMzWBk

## 🎯 LORA OVERVIEW: EDWARD HU
🎯 video by the author EDWARD HU: https://www.youtube.com/watch?v=DhRoTONcyZE&t=34s

LORA
- Technique for fine tuning LLMs
- Between the original model and full fine tuning, everything is LORA in between

![alt text](readme_imgs/HU_Lora_overview1.png)
![alt text](readme_imgs/HU_Lora_overview2.png)
![alt text](readme_imgs/HU_Lora_overview3.png)

- 🎯 Training: there are additional matrices
- 🎯 Inference: LORA updates are additive, no additional matrices. You run inference the same way as the original way. But with LORA inference is much faster
- the checkpoints are much smaller

![alt text](readme_imgs/HU_Lora_details1.png)
![alt text](readme_imgs/HU_Lora_details2.png)
![alt text](readme_imgs/HU_Lora_details3.png)


### 🎯 OTHER ADVANCED IDEAS
- You can cache many LORA models in RAM. When running inference, you can just switch the models between RAM and VRAM (Ram is much larger than VRAM)
- You can also train several LORA modules parallely in a batch, based on the input


## 🎯 LORA DETAILS: MARK HENNINGS
🎯 Youtube video by Mark Hennings: https://www.youtube.com/watch?v=t1caDsMzWBk

### 🎯 TRAINING AN LLM

- PRE TRAINING: Train a base model on a lot of data. like 2 Trillion Tokens
- FINE TUNING-1: pretty much anything you do after on a smaller amount of data 
   - INSTRUCT TUNING: like GPT etc.
   - SAFETY TUNING
   - TASK FINE TUNING
- FINE TUNING-2: pretty much anything you do after on a smaller amount of data 
   - DOMAIN TUNING: like training it for LAW etc..
   - SAFETY TUNING
   - TASK FINE TUNING
![alt text](readme_imgs/hennings_lora1.png)


### 🎯 FULL PARAMETER FINE TUNING
- updates all the model weights. 
- weight matrices are very large. 7B parameter model has 7B weights, 13B parameter model has 13B weights etc
- PROBLEM: all these weights get updated REPEATEDELY IN EVERY EPOCH ! (YEAH 7B updates per epoch ). If you want to go through your entire batch like 100 times, that is 100 eochs. total of 700B updates (WOW !)
- so you can do this only on very very large GPUs



### 🎯 SOLUTION: ENTER LORA
- Lora does not update the weight, it keeps track of only the changes to the weight- DELTA-W)
![alt text](readme_imgs/hennings_lora2.png)

- Instead of having one large matrix with the changes to the weight (DELTA-W ), we use Matrix decomposition to get 2 smaller matrices . These 2 low rank matrices when combined give the DELTA-W matrix
![alt text](readme_imgs/hennings_lora3.png)
- so intead of 5*5 = 25 weight Delta-W matrix, you can use  5*1 and 1*5 low rank matrix . So that is 5+5 = 10 parameters instead of 25
- you can also increase the rank to 5*2 and 2*5 to get a 5*5 matrix. that would be 10+10 = 20 parameters instead of 25. 
![alt text](readme_imgs/hennings_lora4a.png) ![alt text](readme_imgs/hennings_lora4b.png)
- these are toy examples, but you can imagine the gains when you are dealing with a matrix of billions of parameters. 



### RANK in LORA. 
What should the rank be

![alt text](readme_imgs/hennings_lora5.png) ![alt text](readme_imgs/hennings_lora6.png)
![alt text](readme_imgs/hennings_lora7.png) 

### QLORA: QUANTIZED LORA
![alt text](readme_imgs/hennings_lora7.png) 