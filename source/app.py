from flask import Flask, render_template, request, redirect, url_for
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    # Process the query here (e.g., search in a database)
    prompt = f"Generate a recipe for {query}: First, provide the time to prepare, then provide the ingredients needed, and lastly provide numbered preparation steps. Don't include any other information. Use the format: **Time to Prepare:** <time> **Ingredients Needed:** <ingredients> **Preparation Steps:** <steps>. For example, if the recipe is for tacos, it should look like this: **Time to Prepare:** 20–30 minutes **Ingredients Needed:** - 2 large tortillas - 1 cup shredded chicken (or ground beef) - 1 cup chopped onions - 1 cup diced bell peppers - 1 cup chopped tomatoes - 1/4 cup chopped cilantro - 1/2 cup lime juice (or juice and water) - 1/2 cup shredded cheese (grilled or melted) - 1/4 cup chopped green onions - 1/2 cup diced黄瓜 (optional, for extra sweetness) **Preparation Steps:** 1. Heat a pan with 1 tablespoon oil and cook the tortillas until golden brown. 2. Add the shredded chicken and sauté the onions and bell peppers until softened. 3. Add the tomatoes, cilantro, and lime juice to the pan. 4. Mix everything together and spread the mixture on the tortillas. 5. Top each tortilla with cheese and green onions. 6. Serve with additional lime or cilantro for extra flavor."

    text = get_recipe_ai(prompt).split("**")

    time = text[2]
    ingredients = text[4]
    preparation_steps = text[6]
    print("Time to Prepare:", time)
    print("Ingredients Needed:", ingredients)
    print("Preparation Steps:", preparation_steps)
    return render_template('search.html', query=query, time=time,ingredients=ingredients, steps=preparation_steps)

if __name__ == '__main__':
    app.run(debug=True)

def get_recipe_ai(prompt):
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    # prepare the model input
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)

    return content
