from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from optimum.intel.openvino import OVModelForCausalLM
import os

model_id = "meta-llama/Meta-Llama-3-8B"
device = "AUTO:-GPU"

if os.path.exists("ov_model"):
    print("ov_model found.")
    model = OVModelForCausalLM.from_pretrained("ov_model", device=device, verbose=True)
else:
    print("ov_model not found.")
    model = OVModelForCausalLM.from_pretrained(model_id, export=True, device=device)
    model.save_pretrained("ov_model")

tokenizer=AutoTokenizer.from_pretrained(model_id)


pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.float16},
)

k = pipe(
    "How is your day today?",
    max_new_tokens = 128,
    eos_token_id=tokenizer.eos_token_id
  )
print(k)

# Outputted results:

# Using Optimum - generally takes no more than a few seconds to generate after weight compression/compilation
#   CPU: 'generated_text': 'Hey how are you doing today? I am doing well. I am a little bit tired because I'
#   GPU: 'generated_text': 'Hey how are you doing today?defdefQuestion000 (1 ( ( ( ( ( ( (' - consistently nonsense

# Using native transformers class (CPU):
#   Hey how are you doing today? I hope you
#   are doing great. I have a question for you. I’m working on a project and I
#   need a way to detect when a user has clicked on a certain part of the page.
#   Is there a way to do this in JavaScript?\nI’m not sure if you’re familiar
#   with the term “click detection” but it’s basically a way to detect when a
#   user has clicked on a certain part of the page. This can be useful for a
#   variety of things, such as detecting when a user has clicked on a button, or
#   when they have hovered over a certain element.\nThere are a few different
#   ways to do click detection in JavaScript. One way is to use the “on” event
#   listener. This listener will fire when the user clicks on the element that
#   you specify. Another way is to use the “mouseover” event listener. This
#   listener will fire when the user hovers over the element that you
#   specify.\nI hope this helps! Let me know if you have any other
#   questions.\nThanks for reaching out! I’m glad to hear that you’re interested
#   in click detection in JavaScript.\nAs you mentioned, there are a few
#   different ways to do click detection in JavaScript. One way is to use the
#   “on” event listener, which will fire when the user clicks on the element
#   that you specify. Another way is to use the “mouseover” event listener,
#   which will fire when the user hovers over the element that you specify.\nI
#   think the best way to determine which method is best for your project is to
#   try out both methods and see which one works best for your specific needs.
#   You can try out both methods by using the following code:\nvar element =
#   document.getElementById(“myElement”);\nelement.addEventListener(“click”,
#   function() { console.log(“Clicked!”);
#   });\nelement.addEventListener(“mouseover”, function() { console.log(“Hovered
#   over!”); });\nThis code will add a click listener and a mouseover listener
#   to the element with the id “myElement”. When you click on the element, the
#   click listener will fire and the console will log the message “Clicked!”.
#   When you hover over the element, the mouseover listener will fire and the
#   console will log the message “Hovered over!”.\nI hope this helps! Let me
#   know if you have any other questions.\nThanks for your help! I tried out
#   both methods and the “on” event listener worked best for my project. I
#   appreciate your help!
