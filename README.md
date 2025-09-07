# Long-Form AI Video Generation Internship Project

This repository documents my learning journey and project work as an intern exploring the field of long-form AI video generation. The project is structured as a series of tasks, progressing from fundamental concepts of data representation to building a controllable video synthesis pipeline and implementing a generative model from its core principles.

---

## Project Tasks

The project is divided into three main tasks, each building upon the last.

### 📝 Task 1: Foundations of Multimodal Embeddings

**Goal:** To understand how AI models perceive and translate the world of images and text into a mathematical format they can understand.

-   **Part 1: Image Encoding with CNNs:** I started by using a pre-trained Convolutional Neural Network (ResNet50) to convert images into high-dimensional feature vectors. This taught me how complex visual information is distilled into a numerical "fingerprint."

-   **Part 2: Text Encoding with Transformers:** Next, I used pre-trained language models (BERT and CLIP's text encoder) to see how sentences are tokenized and transformed into contextual embeddings. This revealed how models capture the meaning of words based on their surrounding context.

-   **Part 3: Joint Multimodal Embeddings:** Finally, I combined these concepts using CLIP to create a shared embedding space for both images and text. By comparing the cosine similarity between image and text vectors, I validated the model's ability to recognize that an image of a cat is semantically close to the text "a photo of a cat," laying the groundwork for text-guided generation.

### 🎥 Task 2: Controllable Sketch-to-Video Pipeline

**Goal:** To build a practical, end-to-end pipeline that generates a short, controllable video from a sketch and a text prompt.

This task involves creating a workflow in Google Colab that will:
1.  **Input:** Accept a user-drawn sketch and a text description.
2.  **Stylization:** Generate a stylized image from the sketch using **ControlNet** for structural guidance and **LoRA** for artistic style (pokemon).
3.  **Animation:** Animate the stylized image into a 3-5 second video clip using a frame-by-frame generation method.
4.  **Identity Control:** Maintain consistent character identity across all frames, using techniques like **IP-Adapter**
5.  **Dynamic Control:** Allow for scene adaptation by changing the text prompt mid-generation to alter the environment (e.g., from a "sunny forest" to a "stormy night").

### 🔬 Task 3: Implementing a Conditional Diffusion Model

**Goal:** To gain a deep, fundamental understanding of generative modeling by building and training a conditional Denoising Diffusion Probabilistic Model (DDPM).

This task moves beyond using pre-built tools to implementing the core mechanics from scratch. The objectives are to:
1.  **Architecture:** Develop the core components of a conditional DDPM, focusing on the mechanism for injecting conditioning information (like class labels or text embeddings) into the denoising process.
2.  **Training:** Train the model on a small, labeled dataset to learn the reverse (denoising) process, guided by the provided conditions.
3.  **Visualization & Analysis:** Visualize the intermediate steps of the diffusion process to observe how the model gradually reconstructs a clean, coherent image from pure noise based on the input prompt.
4.  **Experimentation:** Explore controllable generation by experimenting with different prompts and conditioning signals to test the model's creative capabilities.

---

## Technologies & Models

-   **Core Libraries:** PyTorch, Transformers (Hugging Face)
-   **Architectures:** CNNs (ResNet), Transformers (BERT, CLIP), DDPMs
-   **Tools & Techniques:** ComfyUI, ControlNet, LoRA, IP-Adapter, Cosine Similarity
-   **Environment:** Google Colab
