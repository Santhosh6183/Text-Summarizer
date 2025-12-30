from transformers import pipeline

# Load summarization model (loads once)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

print("=== Text Summarization AI ===")
print("Enter text to summarize (type 'exit' to quit)\n")

while True:
    text = input("Enter text: ")

    if text.lower() == "exit":
        print("Exiting summarizer...")
        break

    if len(text.split()) < 30:
        print("Please enter at least 30 words for better summary.\n")
        continue

    summary = summarizer(
        text,
        max_length=30,
        min_length=20,
        do_sample=False
    )

    print("\nSummary:")
    print(summary[0]['summary_text'])
    print("-" * 50)
