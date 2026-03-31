Sound Signal Analysis and Gender Classification

This project implements a complete speech analysis pipeline to classify gender and age groups (Male, Female, and Child) from raw audio signals . The system utilizes Fundamental Frequency (F0​) estimation through time-domain analysis techniques.

📂 Directory Structure

To ensure the scripts function correctly, the following directory structure must be maintained:
Plaintext

```text
├── Dataset/                                        # Directory containing raw .wav files
├── create_master_metadata_table.py                 # Script for merging fragmented metadata
├── create_feature_extracted_data.py                # Output containing F0, STE, and ZCR values
├── fft_vs_autocorrelation_graph.py                 # Script for visual signal comparison
├── matrix.py                                       # Script for generating the confusion matrix
├── statistic_and_create_final_performance_table.py # Script for statistical analysis
└── app.py                                          # Streamlit-based user interface
``` 

🛠️ Requirements

The following Python libraries are required for execution:

    Librosa: For audio signal loading and processing.

    Streamlit: For the web-based user interface.

    Scipy & Numpy: For mathematical calculations and Autocorrelation Function (ACF) implementation.

    Pandas: For metadata manipulation and dataset management.

🚀 Execution Order

For correct operation, scripts must be executed in the following sequence:

    create_master_metadata_table.py   
    create_feature_extracted_data.py
    fft_vs_autocorrelation_graph.py 
    matrix.py 
    statistic_and_create_final_performance_table.py 
    app.py 
