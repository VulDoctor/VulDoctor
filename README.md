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
