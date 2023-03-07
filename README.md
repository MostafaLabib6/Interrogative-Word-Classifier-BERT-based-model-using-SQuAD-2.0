# Interrogative-Word-Classifier-BERT-based-model-using-SQuAD-2.0
Here I built a BERT-based model which classifies interrogative words, given question, and passage which includes the answer of the question. For this Exam Generation task, I used the [SQuAD 2.0 dataset](https://rajpurkar.github.io/SQuAD-explorer/). implementation based on [Let Me Know What to Ask: Interrogative-Word-Aware Question
Generation](https://arxiv.org/pdf/1910.13794.pdf)


#### The main steps are:

1. Load the BERT model and tokenizer.
2. Download and preprocess the SQuAD v2 dataset.
3. Tokenize the passages and answers using the tokenizer, and obtain input_ids and attention_masks.
4. Use spaCy to obtain the entity type of each answer and create a learnable embedding.
5. Map each entity to its tensor and create an array for each entity.
6. Define a Feedforward Neural Network (FFN) with a softmax activation function.
7. Map each token in the questions to its integer representation, and store the integer representation of the interrogative words.
8. Run the model with the input_ids, attention_masks, entity_type_embeddings, and interrogative word integer representations as input, and the answers as the target output.


I started with the BERT-base pretrained model [bert-base-uncased](https://huggingface.co/bert-base-uncased) and fine-tune it .
as a benchmark I used the implemented pretrained and fine-tuned model from ***Hugging Face***, named as [BertForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification).



Note: My solution is implemented in PyTorch . For running the notebooks, I used the Google Colab with its GPU.

You can check the Google Colab Notebooks here:
 * InterrogativeWord-Classifier: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BWe5sGX9UjlOnvwY9TSut6Nbofjj5M9z?usp=sharing)

## Update
. train model on the squadV2 whole dataset
. validate it on the squadV2 val dataset
