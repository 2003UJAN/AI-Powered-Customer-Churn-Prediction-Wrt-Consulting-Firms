from google import genai
import os

class GeminiClient:
    def __init__(self, use_vertex=False, project=None, location=None):
        # If you set GOOGLE_API_KEY or ADC beforehand, genai.Client() will use them.
        # For Vertex AI usage set vertexai=True and project/location.
        if use_vertex:
            self.client = genai.Client(vertexai=True, project=project, location=location)
        else:
            self.client = genai.Client()

    def generate_text(self, prompt, model="gemini-2.5-flash", max_output_tokens=256):
        # Simple generation call (synchronous)
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            max_output_tokens=max_output_tokens
        )
        # response object has .text per examples
        return response.text

# quick usage:
# G = GeminiClient()
# out = G.generate_text("Write a short retention message for a churn-risk customer who spends $150/mo.")
# print(out)
