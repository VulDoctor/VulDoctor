# VulDoctor
 
This repository is the replication package of **"Combining Static Purified Semantic and Dynamic Execution Traces for Automated Vulnerability Repair"**.

## Resources

* In the replication, we provide:
  * the scripts we used to:
    * `train.sh`: fine-tune the models with validation.
    * `test.sh`:  perform inference using the fine-tuned models.
  * The source code we used to:
    * `VulDoctor.py`: the main code for training/validating/testing.
    * `datasets/`: training dataset, testing dataset, and valid dataset.
  * dynamic execution trace generation module `DynamicExecutionGeneration/`:
    * `generation.py`: generation of the dynamic execution traces.
    * `fine_tuned_model_epoch10\`: trained model.
  * purification module `purification/`:
    * `slice.py`: get the program slice.

 We use [CodeT5-base](https://drive.google.com/drive/folders/1L5fkJ_J-NvuWlcr-GbfomorxoS6HwuTs?usp=sharing) as our backbone model. Please download the CodeT5-base under the root dir of this replication package. 
 
 In order to replicate the dynamic execution trace generation module, please download the fine_tuned_model for dynamicExecutionGeneration [Here](https://drive.google.com/file/d/1DrkpVKB75a_XK8sjzfl2INhaN_9cCWTJ/view?usp=drive_link) or fine-tune by yourself using the Code [Here](https://github.com/aashishyadavally/nd-slicer). It is worth noted that the dynamicExecutionGeneration module needs to install a new requirements.txt.

 
* `requirements.txt` contains the dependencies needed.

* The experiments were conducted on a server equipped with NVIDIA 3090Ti GPU and Intel Core i7-12700KF, running the Windows.
  
* If you encounter OutOfMemoryError, please note that you typically need around 30 GB of GPU memory to run VulDoctor.

* Other baselines could see [VRepair](https://github.com/ASSERT-KTH/VRepair), [VulRepair](https://github.com/awsm-research/VulRepair), [VulMaster](https://github.com/soarsmu/VulMaster_), and [VQM](https://github.com/awsm-research/VQM)

## Install dependencies

 Please install them first.
```
conda create -n VulDoctor python=3.8 
conda activate VulDoctor
pip install -r requirements.txt
```

## Train and Test 

To replicate VulMaster, ensure that `datasets/` is in the root path of this project. 

Inference:
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

Training:
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
