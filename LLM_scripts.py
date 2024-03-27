from transformers import GPT2Tokenizer, GPT2LMHeadModel
print('libraries imported')

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
print('tokenizer imported')

model = GPT2LMHeadModel.from_pretrained('gpt2')

print('model imported')




def chat(user_input):
    # Tokenize the user input and convert to tensor
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    # Generate a response from the model
    chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=2, top_p=0.92, top_k=50)
    
    # Decode and print the model's response
    chat_output = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return chat_output
print('function defined')




print("Chatbot initialized. Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    response = chat(user_input)
    print("Chatbot:", response)






