# VulDoctor
 
This repository is the replication package of **"Combining Static Purified Semantic and Dynamic Execution Traces for Automated Vulnerability Repair"**.

## Resources

* In the replication, we provide:
  * the scripts we used to:
    * `0a_train.sh`: fine-tune the models with validation.
    * `0b_test.sh`:  perform inference using the fine-tuned models.
  * the source code we used to:
    * `VulDoctor.py`: the main code for training/validating.
    * `infere.py`: the main code for testing.
    * `src/`: the utils, model, data preprocessing, etc. for FiD.
  * the CWE knowledge we collected from CWE website:
    * `CWE_examples_GPT35_generate_fixes_full.csv`: the raw vulnerable code examples and their analysis and CWE names directly collected from CWE homepages.
    * `chatgpt_api_generate_fix.py`: the main code for generating fixes for vulnerable code examples with expert analysis as guidance.
    * `ChatGPT_generated_fixes_labels.xlsx`: the manually labeled correctness for generated fixes for vulnerable code examples.

 We stored the datasets you need in order to replicate our experiments at: https://zenodo.org/records/10150013 and [Here](https://drive.google.com/drive/folders/1L5fkJ_J-NvuWlcr-GbfomorxoS6HwuTs?usp=sharing) is CodeT5 model after adaptation. 
 
* `requirements.txt` contains the dependencies needed.

* The experiments were conducted on a server equipped with NVIDIA L40 GPU and Intel(R) Xeon(R) CPU E5-2420 v2@ 2.20GHz, running the Ubuntu OS.
  
* If you meet OutOfMemoryError: please note that you typically need around 30 GB GPU memory to run VulMaster.

## Install dependencies

 Please install them first.
```
conda create -n VulDoctor python=3.8 
conda activate VulDoctor
pip install -r requirements.txt
```

## Train and Test 

To replicate VulMaster, ensure that `c_dataset/` is in the root path of this project. 

Training:
```
python VulDoctor.py \
    --output_dir=./saved_models \
    --model_name=model.bin \
    --tokenizer_name=Salesforce/codet5-base \
    --model_name_or_path=Salesforce/codet5-base \
    --do_test \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --num_beams=50 \
    --eval_batch_size 1
```

Testing:
```
python VulDoctor.py \
    --model_name=model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=Salesforce/codet5-base \
    --model_name_or_path=Salesforce/codet5-base \
    --do_train \
    --epochs 75 \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
```
