import csv
import re

# Input and output file paths
input_file = '/home/vansh/Documents/PROJECT/PROJECT_CLARK/questions_dataset/Java_MCQ.txt'
output_file = '/home/vansh/Documents/PROJECT/PROJECT_CLARK/questions_dataset/Java_MCQ.csv'


# Open the input file for reading
with open(input_file, 'r') as infile:
    lines = infile.readlines()

# Initialize variables
questions = []
current_question = {"question": "", "answer": ""}

# Process the lines
for line in lines:
    line = line.strip()
    if line.startswith("Ans."):
        current_question["answer"] = line.replace("Ans. ", "")
        questions.append(current_question)
        current_question = {"question": "", "answer": ""}
    else:
        current_question["question"] += line

# Write questions and answers to CSV
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['question', 'answer']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for question in questions:
        writer.writerow(question)

print(f'Conversion complete. CSV file saved as {output_file}')
