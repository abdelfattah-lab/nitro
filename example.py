from models.llama import Llama

llama = Llama.from_pretrained(pretrained_model="meta-llama/Meta-Llama-3-8B",
                                model_dir="llama_npu",
                                max_batch_size=1, max_seq_len=256, chunk_size=16,
                                export=True,
                                device="NPU",
                                compile=True,
                                compress=False,
                                verbose=True)

output = llama.generate(prompt=["I was wondering what is going on"],
                        max_new_tokens=10)

print(output)