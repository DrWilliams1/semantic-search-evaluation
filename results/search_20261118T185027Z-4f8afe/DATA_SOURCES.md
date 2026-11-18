# Data and code sources

This retrieval experiment extends Chapter 8 of Hands-On Large Language Models by Jay Alammar and Maarten Grootendorst (2024): https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter08 . Experimental code has been substantially changed; see the retained Apache-2.0 LICENSE.

Query/document IDs and relevance judgments are derived from BEIR SciFact: https://huggingface.co/datasets/BeIR/scifact and https://huggingface.co/datasets/BeIR/scifact-qrels . These data-derived records retain CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/ . Judgments are unchanged; development query records are a selected subset. Source revisions and file hashes are recorded in config.json and data_record.json. Corpus text and model weights are not included.

References: Thakur et al. (2021), BEIR, https://arxiv.org/abs/2104.08663 ; Wadden et al. (2020), Fact or Fiction: Verifying Scientific Claims, https://aclanthology.org/2020.emnlp-main.609/ .
