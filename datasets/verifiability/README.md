# Merged Verifiability Dataset

This is a combination of the dataset used by Park and Cardie (2014), containing
sentences from comments on Regulation Room, and a dataset created at the
University of Groningen in 2022, containing sentences from the Subreddit _Change
My View_.

The following fields are available:

- `sentence` – the sentence, extracted from a comment, that was annotated
- `thread_id` – An identifier for the thread or rule the comment was placed in
- `comment_id` – An identifier for the comment in the thread
- `sentence_index` – a serial number indicating at which position the sentence
  occurred in the original comment. I.e., an index of 3 means this was the
  fourth sentence in the comment.
- `source` – Whether the data came from Regulation Room or Change My View
- `verifiability` – One of (verifiable, unverifiable, nonargument)
- `experiential` – One of (True, False, N/A); experientiality only applies to
  verifiable sentences.

---

Park, J., & Cardie, C. (2014). Identifying Appropriate Support for Propositions
in Online User Comments. Proceedings of the First Workshop on Argumentation
Mining, 29–38. <https://doi.org/10/gg29gq>
