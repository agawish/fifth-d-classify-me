import json

import redis


class RedisPromptCache:
    def __init__(self, redis_url: str):
        self.redis = redis.Redis.from_url(redis_url)

    def add_prompt(self, prompt_key: str, response: str, expiry: int = 300):
        self.redis.set(prompt_key, response, ex=expiry)

    def get_prompt(self, prompt_key: str) -> str:
        cached_response = self.redis.get(prompt_key)
        if cached_response:
            return json.loads(cached_response)
        return ""
