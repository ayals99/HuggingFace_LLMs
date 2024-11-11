from transformers import pipeline
from pandas import DataFrame

# Start of Sentiment analysis.
sentimentClassifier = pipeline("sentiment-analysis")
foodSentenceList = ["I don't like the food",
                "Although the food here is not the best, this place has some good vibes",
                "The stay was so-so"]

sentimentResults = sentimentClassifier(foodSentenceList)

print("Sentiment results are: ")

for i in range(len(sentimentResults)):
    result = sentimentResults[i]
    sentence = foodSentenceList[i]
    print('"', sentence, '"', "is", result['label'], "with certainty of %",result['score']*100)


# End of Sentiment analysis.

# Start of Custom Label classification:
labelOptions = ["winter", "law enforcement", "disney movies", "education"]
customLabelClassifier = pipeline("zero-shot-classification" , candidate_labels = labelOptions)
topicsSentenceList = ["Do you want to build a snowman?",
                      "Freeze, you're under arrest",
                      "The ice age was twenty thousand years ago and spanned over the whole world"] 
customLabelResult = customLabelClassifier(topicsSentenceList)
# print(customLabelResult)

# Helper function to receive a non-empty list 
def findMaxValIndices(lst: list)->list:
    maxVal = max(lst)
    result = [index for index, val in enumerate(lst) if val == maxVal]
    return result

def printTopicAnswer(sentence:str, category: str, certainty: int):
    print('"{}" is about {}. Certainty: %{}'.format(sentence, category, certainty*100))

print("Categorizing results are: ")
for result in customLabelResult:
    maxIndices = findMaxValIndices(result['scores'])    
    for i in range(len(maxIndices)):
        if i > 0:
            print("OR")
        printTopicAnswer(result['sequence'], labelOptions[maxIndices[i]], result['scores'][maxIndices[i]])

# End of Custom Label classification:

# Start of Text Generation:
textGenerator = pipeline("text-generation")
generatedResult = textGenerator("Write me the best beef bourguignon recipe")
print("Text Generator Result is:")
print(generatedResult)

# End of Text Generation:

# Start of Question answering:
QuestionAnswerer = pipeline("question-answering")

QuestionAnswer = QuestionAnswerer(question = "How tall is the Eifel Tower in meters",
                                  context = "The Eifel tower is 330 metres (1,083 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest human-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure in the world to surpass both the 200-metre and 300-metre mark in height. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.")

print("The height of the Eifel Tower (in meters) is: {}".format(QuestionAnswer['answer']))
