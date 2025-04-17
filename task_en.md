# Advanced Machine Learning Techniques
## PART THREE: FINE TUNING (intermediate)

### Requirements:
In this practical exercise, you will perform fine-tuning of a Large Language Model (LLM). The requirements are:

* Use a small LLM (1B, 2B, 3B, or 4B parameters, max 8B), for example Llama 3.2 (1B).
* Use Python for implementation.
* Perform fine-tuning locally on your machine.
* Recommended libraries:
  * **Torch (PyTorch)**: The fundamental library for working with tensors
  * **PEFT (Parameter-Efficient Fine-Tuning)**: Provides methods like LoRA that allow efficient training of only a small portion of parameters in a large model
  * **BitsandBytes**: Enables quantization, which reduces the model's memory consumption on GPU
* **CUDA** allows training using GPU (though CPU training is also possible).
* Use the provided knowledge document about waste management in Olot.
* Use English for all interactions with the Chat.

### Objectives:
* Create code to perform local fine-tuning of a small LLM based on the provided knowledge document.
* Train the LLM to learn the information from the supplied knowledge document.
* Calculate the performance of the LLM on the entire dataset of questions from the knowledge document before training, using GPTScore as the metric.
* Compare the fine-tuned LLM using the same indicator on the entire dataset of questions from the knowledge document after training to observe the improvement.

### Deliverables:
* Description of the process: selected LLM, pre-processing steps, configuration, and main hyperparameters.
* Functional and commented code with prerequisites.
* Questions, answers, and GPTScore before fine-tuning.
* Questions, answers, and GPTScore after training.

### Notes:
* You may add other metrics in addition to GPTScore if desired.
* Submit everything in a zip or rar file.