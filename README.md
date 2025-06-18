![image](https://github.com/user-attachments/assets/38827a5f-2704-4a3e-ba77-86aab014e9f2)


The Architecture of EDCT-BinFormer**

### 1.Data Splitting

Specify the data path, split size, validation, and testing sets to prepare your data. In this example, we set the split size as (256 X 256), the validation set as 2016, and the testing set as 2018 while running the process_dibco.py file.

```bash
python process_dibco.py --data_path /YOUR_DATA_PATH/ --split_size 256 --testing_dataset 2018 --validation_dataset 2016
```

## EDCT-BinFormer
### 2.Training
For training, specify the desired settings (batch_size, patch_size, model_size, split_size, and training epochs) when running the file train.py. For example, for a base model with a patch size of (16 X 16) and a batch size of 32, we use the following command:

```bash
python train.py --data_path /YOUR_DATA_PATH/ --batch_size 32 --vit_model_size base --vit_patch_size 16 --epochs 151 --split_size 256 --validation_dataset 2016
```
You will get visualization results from the validation dataset on each epoch in a folder named vis+"YOUR_EXPERIMENT_SETTINGS" (it will be created). In the previous case, it will be named visbase_256_16. Also, the best weights will be saved in the folder named "weights".

### 3. Testing on a DIBCO dataset
To test the trained model on a specific DIBCO dataset (should match the one specified in Section Process Data, if not, run process_dibco.py again). Use your own trained model weights. Then, run the below command. Here, I test on H-DIBCO 2017, using the base model with a 16X16 patch size and a batch size of 16. The binarized images will be in the folder ./vis+"YOUR_CONFIGS_HERE"/epoch_testing/ 
```bash
python test.py --data_path /YOUR_DATA_PATH/ --model_weights_path  /THE_MODEL_WEIGHTS_PATH/  --batch_size 16 --vit_model_size base --vit_patch_size 16 --split_size 256 --testing_dataset 2017
```



## Acknowledgement

Our project has adapted and borrowed the code structure from [T2T-Binformer]((https://github.com/RisabBiswas/T2T-BinFormer)). We are thankful to the authors! 


## Citation

If you use the T2T-BinFormer code in your research, we would appreciate a citation to the original paper:
```
  @misc{biswas2023layerwise,
        title={A Layer-Wise Tokens-to-Token Transformer Network for Improved Historical Document Image Enhancement}, 
        author={Risab Biswas and Swalpa Kumar Roy and Umapada Pal},
        year={2023},
        eprint={2312.03946},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
  }
```
