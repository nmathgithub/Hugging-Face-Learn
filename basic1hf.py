from transformers import pipeline
classifier = pipeline("sentiment-analysis")
res = classifier("Hope is a good thing, maybe the best of things. And no thing ever dies.")

print(res)

# Add a comment for push
# Tutorial Source: https://www.youtube.com/watch?v=QEaBAZQCtwE