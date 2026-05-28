# %%
import mlflow
import numpy as np
from dotenv import load_dotenv

load_dotenv(override=True)
# %%
import polars as pl

data = pl.read_parquet("https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet")

data.head()
data.shape
# %%
n_classes = len(data['code'].unique())
n_classes

# %%
from sklearn.model_selection import train_test_split

X = data['label'].to_numpy()
y = data['code'].to_numpy()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=99)

X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=0.5, random_state=99)

# %%
print("X_train: ", len(X_train))
print("X_valid: ", len(X_valid))
print("X_test: ", len(X_test))

print("y_train: ", len(y_train))
print("y_valid: ", len(y_valid))
print("y_test: ", len(y_test))

# %%
# Check all labels in training set
print("y_train labels: ", len(np.unique(y_train)))

# %%
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(y_train)

y_train_enc = encoder.transform(y_train)
y_valid_enc = encoder.transform(y_valid)
y_test_enc = encoder.transform(y_test)

# %%
from torchTextClassifiers.value_encoder import ValueEncoder

value_encoder = ValueEncoder(encoder)

# %%
from torchTextClassifiers.tokenizers import WordPieceTokenizer

tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=10)

tokenizer.train(X_train)
# %%
example = X_train[40070]
print("Example label: ", example)

example_tokenize = tokenizer.tokenize(example).input_ids.squeeze(0)

tokenizer.tokenizer.convert_ids_to_tokens(example_tokenize)

# %%
from torchTextClassifiers.torchTextClassifiers import ModelConfig, torchTextClassifiers, TrainingConfig

config = ModelConfig(embedding_dim=96, num_classes=n_classes)

# %%
classifier = torchTextClassifiers(tokenizer=tokenizer, model_config=config, value_encoder=value_encoder, )

# %%
training_config = TrainingConfig(lr=5e-4,
                                batch_size=128,
                                num_epochs=2)
# %%
mlflow.set_experiment("nace_code_test")
mlflow.pytorch.autolog()

with mlflow.start_run():
    classifier.train(X_train, y_train, 
                    training_config=training_config,
                    X_val=X_valid, y_val=y_valid,
                    verbose=True)


    mlflow.log_artifacts(
        training_config.save_path,   # local folder produced by ttc.train()
        artifact_path="model_artifacts",
    )

# %%
# Get pretrained model
import s3fs

fs = s3fs.S3FileSystem(
    anon=True,  # public bucket
    endpoint_url="https://minio.lab.sspcloud.fr",
)

local_dir = "./mlflow-artifacts/"
fs.get(
    "projet-funathon/diffusion/mlflow-artifacts/",
    local_dir,
    recursive=True,
)
# Rebuild the torchTextClassifiers object from the downloaded files
ttc = torchTextClassifiers.load(local_dir)

ttc.pytorch_model.eval()

# %%
import random
top_k = 5

random_indices = random.sample(range(len(X_test)), 5)
example_texts = X_test[random_indices]
example_true_codes = y_test[random_indices]

explanation = ttc.predict(example_texts, top_k=5, explain_with_captum=True)

#%%
print("Examples: ", example_texts)

for i, text in enumerate(example_texts):
    print("True code: ", example_true_codes[i])
    print("Prediction: ", [explanation['prediction'][i][k] for k in range(top_k)])
    print("Confidence: ", [explanation['confidence'][i][k].item() for k in range(top_k)])
    #print("Attributions: ", [explanation['confidence'][i][k] for k in range(top_k)])


# %%
captum_attributions = explanation['captum_attributions']

from torchTextClassifiers.utilities.plot_explainability import (
    map_attributions_to_char, map_attributions_to_word,
    plot_attributions_at_char, plot_attributions_at_word, figshow,
)
text_idx = 0
top_k_idx = 2
text_sample         = example_texts[text_idx]
offsets             = explanation["offset_mapping"][text_idx]
word_ids            = explanation["word_ids"][text_idx]
predicted_code = explanation["prediction"][text_idx][top_k_idx]

attributions  = explanation["captum_attributions"][text_idx][top_k_idx] # (seq_len,)

words, word_attributions = map_attributions_to_word(
    attributions.unsqueeze(0), text_sample, word_ids, offsets
)
char_attributions = map_attributions_to_char(attributions.unsqueeze(0), offsets, text_sample)

titles = [f"Attributions for NACE code {predicted_code}"]

figshow(plot_attributions_at_char(
    text=text_sample, attributions_per_char=char_attributions, titles=titles,
)[0])

figshow(plot_attributions_at_word(
    text=text_sample, words=words.values(), attributions_per_word=word_attributions, titles=titles,
)[0])
# %%
explanation.keys()
# %%
results_test = ttc.predict(X_test, top_k=1)

preds    = results_test["prediction"].squeeze(1)
accuracy = (preds == y_test).mean()
print(f"Test accuracy: {accuracy:.4f} ({int(accuracy * len(y_test))}/{len(y_test)} correct)")
# %%
