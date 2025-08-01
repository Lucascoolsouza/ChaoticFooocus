Usage inside Fooocus
1. Instantiate once (global):
Python
Copy
from vibe_memory import VibeMemory
vibe = VibeMemory("memory.json")   # auto-loads or creates
2. Filter generations:
Python
Copy
from vibe_memory import apply_vibe_filter

latent = apply_vibe_filter(
    latent_dict=latent,
    vae=vae,
    clip_model=vibe.clip_model,  # reuse same CLIP
    vibe=vibe,
    threshold=0.0,      # reject anything below neutral
    max_retry=3,
    async_task=async_task
)
3. UI buttons (pseudo-code):
Python
Copy
# On "👍" click:
like_current(latent, vae, vibe)

# On "👎" click:
dislike_current(latent, vae, vibe)
🌟 Optional Aesthetic Predictor Swap
Replace image_to_embedding with a small aesthetic classifier (e.g. aesthetic-predictor-v2):
Python
Copy
from transformers import pipeline
aesthetic = pipeline("image-classification",
                     model="shadowlilac/aesthetic-predictor")

def aesthetic_embedding(self, pil_image):
    label_score = aesthetic(pil_image)[0]  # {'label': 'good', 'score': 0.93}
    # Convert to vector: [score_good, score_bad]
    vec = [label_score['score'] if label_score['label']=='good' else 1-label_score['score'],
           label_score['score'] if label_score['label']=='bad' else 1-label_score['score']]
    return vec
Now your JSON can hold vectors like [0.93, 0.07] and still use cosine similarity.
🧠 Memory.json example
JSON
Copy
{
  "liked": [
    [0.123, -0.456, ...],
    [0.111, 0.222, ...]
  ],
  "disliked": [
    [-0.333, 0.999, ...]
  ]
}
No paths needed if you only store vectors.