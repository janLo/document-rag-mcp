import base64
from openai import AsyncOpenAI
from ..config import VisionConfig


class VisionClient:
    def __init__(self, config: VisionConfig):
        self.config = config
        self.client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
        self.model = config.model

    async def extract_text_from_image(self, image_bytes: bytes) -> str:
        """Extracts text content from a page image using an OpenAI-compatible vision model.

        Returns an empty string if vision is disabled.
        """
        if not self.config.enabled:
            return ""

        # Encode image to base64 string
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract all text content from this document page image. "
                            "Preserve layout structure, headings, and tabular data from charts or tables. "
                            "Do not wrap the output in markdown code blocks like ```txt or ```md. "
                            "Only output the raw text extracted from the document."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                ],
            }
        ]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2048,
            temperature=0.0,
        )

        return response.choices[0].message.content or ""
