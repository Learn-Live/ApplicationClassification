<!--- comment 
> ## 'data' folder
>>>--- raw data (pcap)
--->


# Purpose
    "To identify pkts or flow progressively".

## 1. Requirements
   - python3 > 3.6
   - pytorch > 0.4
   - numpy
   - matplotlib
   - sklearn


## 2. Project Directory Structure
### |- input_data: raw data
    'if any data is more than 100MB, please do not store it at here'
    data/Wednesday-workingHours-withoutInfinity-Sampled.pcap_ISCX.csv

### |- output_data: results
    ...
    
### |- log: use to log middle or tmp results.
    ...

### |- proposed_algorithms
    ### |- deep_autoencoder_pytorch
            main_autoencoder.py

### |- compared_algorithms
    ### |- DT_Sklearn
        main_DT.py
        basic_svm.py

### |- utilities
    CSV_Dataloder.py
    common_funcs.py
    
    ## 'pcap2flow' folder
    >>>--- toolkit to convert pcap files to txt or feature data.
    
    ## 'preprocess' folder 
    >>>--- toolkit to preprocess input data, such as 'load data', 'normalization data'
        
    ## |- visualization: plot data to visualize 
        ..
    

### |-history_files: backup 
    ...

## Note:
    since 10/13, ...
