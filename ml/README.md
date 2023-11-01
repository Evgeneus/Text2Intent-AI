# ML Research Part
This directory represents ML functionality, in particular research part, of the intent-classifier service.
- `data` folder contains train-test datasets and supplementary data files necessary for training models.
- `notebooks` folder contains notebooks for data exploration and model training, all of them are available on google colab (links can be found below).

## Notebooks
>### data_exploration_transformation.ipynb
This notebook is dedicated to exploratory analyses of the ATIS dataset. Some key observations:

1. Highly imbalanced dataset!
2. `time` representation in the dataset has different formats.
3. Looks like the text is lowercased -> need to take this into account at inference time in prod to avoid data shift. 
4. No duplicates in the datasets, however, there is data leakage between train and test sets.
5. There is some label mismatch in train and test sets.
6. Some labels are represented as a combination of several other intents (labels) from the dataset.
7. Looks like punctuation was removed -> need to take this into account at inference time in prod.
8. The text length is relatively short -> we may think about reducing the sequence length in Transformer for faster inference time (future work).
9. Some labels are not clean.
10. Some classes are duplicated, e.g., "flight+airfare" and "airfare+flight" in the test set, "flight_no+airline" to "airline+flight_no" in both sets.

---> The dataset needs cleaning and transformation.

>### atis_mpnet.ipynb

This notebook showcases a PyTorch-based code for fine-tuning transformer models designed for sequence classification.
The code can be easily used for various classification tasks, working on a CPU, a single GPU, or multiple GPUs using Distributed Data Parallel (DDP). The code allows you to train and fine-tune any sequence classification model available in the HuggingFace repository.

This implementation leverages the capabilities of both PyTorch and PyTorch Lightning deep learning frameworks, ensuring robust multi-GPU training.

Throughout the training process, I monitor and compute running precision, recall, and accuracy for each class, providing insights into model performance during the training phase. After the training stage concludes, an evaluation on the test dataset is performed, resulting in a metrics report computed with sklearn. The report includes an expected calibration error score (ECE) and a reliability diagram. Furthermore, the predictions and associated probabilities are saved in a CSV file for further analysis.

To ensure compatibility and accessibility, the model is exported in both torch and ONNX formats, making it easy to integrate and deploy in a variety of settings.

>### atis_setfit.ipynb
I am curious to read new AI papers and try out new approaches, for this challenge, I have also investigated a recent work "Efficient Few-Shot Learning Without Prompts" (SetFit) by Tunstall et al. (https://arxiv.org/pdf/2209.11055.pdf).

SetFit is a new approach to text classification that involves fine-tuning a Sentence Transformer with task-specific data. It offers a simpler and more efficient alternative to few-shot classification training. So first the approach learns embeddings of texts in a Siamese and Contrastive learning fashion, and then we can tune a one-layer classifier on top of the frozen encoder. Thus I have researched this approach for the ATIS challenge and found it very interesting to consider for the future (at least because the setfit codebase is very limited). Moreover, we can consider enrolling new classes without retraining the network, in particular using KNN search in the reference database of embeddings.

>### prod_monitoring.ipynb
This notebook simulates the monitoring of a model operating in a production environment, where there is no immediate ground truth data.

## Experiment Takeaways
### I have tuned several multilingual transformer models including: 
- `xlm-align-base` from Microsoft, 280 M parameters. I have found this model to be a very good baseline for many tasks. 
- `all-MiniLM-L12-v2` - is a small sentence-transformer that was pre-trained to represent the semantics of a text snippet in a single embedding vector, 33.4M params.
- `all-mpnet-base-v2` from Microsoft - is another sentence-transformer, 109M params.

When choosing a model for deployment in a production environment, we need to go beyond just assessing its performance metrics. In addition to evaluating accuracy, we must consider factors like inference time and the model's adaptability for scope extension, for instance the seamless addition of new classes.

Furthermore, a crucial model characteristic is its ability to accurately estimate the likelihood of its predictions, which is essentially a measure of its calibration. When a model is well-calibrated, we can confidently rely on its predictions when it expresses high certainty, and conversely when it's less sure. This aspect becomes particularly pivotal when evaluating a model in a production environment where ground truth labels are absent, as performance metrics are computed based on predicted probabilities. To assess model calibration, I will utilize Expected Calibration Error (ECE) and the Reliability Diagram.


_Final Test metrics:_
- model: all-mpnet-base-v2
- size: 420 MB, 109M params
- ECE score: 0.0245
- accuracy: 0.9778
- f1-macro: 0.7334

|                        | Precision | Recall | F1-Score | Support |
|------------------------|-----------|--------|----------|---------|
| abbreviation           | 1.00      | 0.96   | 0.98     | 27      |
| aircraft               | 0.88      | 0.88   | 0.88     | 8       |
| aircraft+flight+flight_no | 0.00   | 0.00   | 0.00     | 0      |
| airfare                | 1.00      | 0.91   | 0.95     | 53      |
| airfare+flight_time    | 0.00      | 0.00   | 0.00     | 0       |
| airline                | 1.00      | 1.00   | 1.00     | 28      |
| airline+flight_no     | 1.00      | 1.00   | 1.00     | 1       |
| airport                | 1.00      | 0.93   | 0.96     | 14      |
| capacity               | 1.00      | 1.00   | 1.00     | 21      |
| cheapest               | 0.00      | 0.00   | 0.00     | 0       |
| city                   | 1.00      | 1.00   | 1.00     | 5       |
| day_name               | 1.00      | 1.00   | 1.00     | 2       |
| distance               | 1.00      | 1.00   | 1.00     | 10      |
| flight                 | 0.99      | 1.00   | 0.99     | 609     |
| flight+airfare         | 0.62      | 0.80   | 0.70     | 10      |
| flight+airline         | 0.00      | 0.00   | 0.00     | 0       |
| flight_no              | 1.00      | 1.00   | 1.00     | 8       |
| flight_time            | 1.00      | 1.00   | 1.00     | 1       |
| ground_fare            | 1.00      | 0.88   | 0.93     | 8       |
| ground_service         | 1.00      | 1.00   | 1.00     | 36      |
| ground_service+ground_fare | 0.00   | 0.00   | 0.00 | 0      |
| meal                   | 1.00      | 1.00   | 1.00     | 6       |
| quantity               | 1.00      | 0.38   | 0.55     | 8       |
| restriction            | 0.50      | 1.00   | 0.67     | 1  
| -            | -      | -  | -     | -  
|
| accuracy               |           |        | 0.98     | 856     |
| macro avg              | 0.75      | 0.74   | 0.73     | 856     |
| weighted avg           | 0.99      | 0.98   | 0.98     | 856     |

## Some Notes about SetFit Experiments

By employing SetFit, I managed to achieve comparable metrics to those mentioned above using a more compact model. I utilized the `all-MiniLM-L12-v2` model, which has the following specifications:

**Model Specification**:
- Model Name: all-MiniLM-L12-v2
- Number of Parameters: 33.4 million
- Max Sequence Length: 256 (Suitable for Atis)
- Model Size: 120 MB

Remarkably, this allowed me to train the end-to-end network in a few-shot regime in less than **1 minute of GPU time**, as detailed in the notebook.

The outcomes were impressive:

**Metrics**:
- ECE (Expected Calibration Error): 0.0276
- Accuracy: 0.9707
- F1-macro: 0.79

These results demonstrate the efficiency and effectiveness of this approach.
