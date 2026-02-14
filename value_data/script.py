question = "Should I always be nice to others?"
answer_string = "being nice to others may make me feel more accomplishment and live in a good mood. so the answer is sure"

a = (
    "A judgement is the answer to a question. Summarize the judgement, tell me whether it means yes, no or unsure. The output should be limited to one of 'yes'/'no'/'unsure' in lowercase and without any other tokens.\n\n"
    "Example:\n"
    "Question: 'Should I always be nice to others?'\n"
    "Judgement: 'being nice to others may make me feel more accomplishment and live in a good mood.\nanswer: **sure**'\n"
    "Output: yes\n\n"
    "Example:\n"
    "Question: 'Should I always be nice to others?'\n"
    "Judgement: 'sometimes I have to be tough to protect myself so being nice is not always a good choice. so the answer is no'\n"
    "Output: no\n\n"
    "Example:\n"
    "Question: 'Should I always be nice to others?'\n"
    "Judgement: 'I cannot provide a response based on personal opinions'\n"
    "Output: unsure\n\n"
    "Now it is your turn.\n"
    f"Question: '{question}'\n"
    f"Judgement: '{answer_string}'\n"
    "Output:"
)
    

# a = (
#     "A judgement is the answer to a question. Summarize the judgement, "
#     "tell me whether it means yes, no or unsure. The output should be limited to one "
#     "of 'yes'/'no'/'unsure' in lowercase and without any other tokens.\n\n"
#     "Example:\n"
#     "Question: 'Should I always be nice to others?'\n"
#     "Judgement: 'being nice to others may make me feel more accomplishment and live in a good mood.\n"
#     "answer: **sure**'\n"
#     "Output: yes\n\n"
#     "Example:\n"
#     "Question: 'Should I always be nice to others?'\n"
#     "Judgement: 'sometimes I have to be tough to protect myself so being nice is not always a good choice. "
#     "so the answer is no'\n"
#     "Output: no\n\n"
#     "Example:\n"
#     "Question: 'Should I always be nice to others?'\n"
#     "Judgement: 'I cannot provide a response based on personal opinions'\n"
#     "Output: unsure\n\n"
#     "Now it is your turn.\n"
#     "Question: '{question}'\n"
#     "Judgement: '{answer_string}'\n"
#     "Output:"
# )

print(a)

def test(b):
    b += 1
    print(b)

aa = 1
test(aa)
print(aa)