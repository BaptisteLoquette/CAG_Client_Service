# CAG_Client_Service

This repo exposes to perform, from simple, Cache Augmented Generation (CAG) **with small models** (Qwen2.5-0.5B) or any model.
To show that **small models** can still **be usefull on scalable** (out-of-domain) **Augmented Generation**.

CAG is more or less **40x faster than RAG**__, and needs no pre-indexing !
Further more the **Augmented generation is done in one pass** as the knowledge base is encoded and pre-loaded in the kv-cache.

[video_CAG_tolk_ai_github_README.webm](https://github.com/user-attachments/assets/353b0bfb-32dd-45fe-995a-22c97db9d050)
