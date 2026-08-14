from cachetools import TTLCache

from grox.core.tasks.task_rate_limit import TaskTTLDedupeWithPost


class TaskRateLimitEmbeddingV5(TaskTTLDedupeWithPost):
    DEDUPE_CACHE = TTLCache(maxsize=10_000, ttl=60)
    DEDUPE_NAME = "mm emb v5"


class TaskRateLimitEmbeddingV5ForReply(TaskTTLDedupeWithPost):
    DEDUPE_CACHE = TTLCache(maxsize=10_000, ttl=60)
    DEDUPE_NAME = "mm emb v5 for reply"


class TaskRateLimitEmbeddingV82(TaskTTLDedupeWithPost):
    DEDUPE_CACHE = TTLCache(maxsize=10_000, ttl=60)
    DEDUPE_NAME = "mm emb v8.2"


class TaskRateLimitEmbeddingV82ForReply(TaskTTLDedupeWithPost):
    DEDUPE_CACHE = TTLCache(maxsize=10_000, ttl=60)
    DEDUPE_NAME = "mm emb v8.2 for reply"
