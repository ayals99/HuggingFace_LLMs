from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

Sentences = ["To be or not to be", "That is the question"]

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer)


result = tokenizer(Sentences, padding = True, truncation=True, return_tensors="pt")

print("Result is: {}".format(result))

# Using the Model to turn the tokens into Logits:
print()
outputs = model(**result)
print("The size of the output array, after using the model, is: {}".format(outputs.logits.shape))

print("And the Logits that the model computed are: {}, {} for the first sentence and {}, {} for the second"\
      .format(outputs.logits[0][0],outputs.logits[0][1], outputs.logits[1][0], outputs.logits[1][1]) )


# Using the Softamx function on each of the two 2*1 vector to get the "Probabilities":
print()
probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
print("The probabilites are: {} {}, {} {} and {} {}, {} {}".format(probabilities[0][0], model.config.id2label[0],\
                                                        probabilities[0][1],model.config.id2label[1],\
                                                        probabilities[1][0], model.config.id2label[0],\
                                                        probabilities[1][1],model.config.id2label[1]))

# Diving into what the Tokenizer does:
print()
print("NOTE: Now I'll dive into the tokenizer and take a look at what it does in each step:")

print()
tokens = tokenizer.tokenize(Sentences)
print("Tokens list is: {}".format(tokens))

print()
IDs = tokenizer.convert_tokens_to_ids(tokens)
print("IDs of the tokens are: {}".format(IDs))

print()
decodedSentence = tokenizer.decode(IDs)
print("Decoded Sentence is: {}".format(decodedSentence))

print()
print("Now we'll pass the IDs to the model as if they're one sentence:")
inputIDs = torch.tensor([IDs])
output = model(inputIDs)
print("Resulting Logits are:", output.logits)

# Using an Attention mask:
print()
print("Now we'll pass the IDs to the model with an Attention mask:")
AttentionMask = [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 0, 0]]

# Getting the Ids of the tokens for each sentence individually:
firstSentence = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(Sentences[0]))
secondSentence = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(Sentences[1]))

# Padding secondSentence to be the same length as firstSentence:
for i in range(len(firstSentence) - len(secondSentence)):
    secondSentence.append(tokenizer.pad_token_id)

batchedInput = [firstSentence, secondSentence]
print("The final result of Logits with an Attention Mask is:")
print(model(torch.tensor(batchedInput), attention_mask = torch.tensor(AttentionMask)))