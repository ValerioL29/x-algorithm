from cachetools import TTLCache

from grox.core.tasks.task_rate_limit import TaskTTLDedupeWithPost


class TaskRateLimitReplySpamAnnotationWithPost(TaskTTLDedupeWithPost):
    DEDUPE_CACHE = TTLCache(maxsize=10_000, ttl=60)
    DEDUPE_NAME = "reply spam"


class TaskRateLimitReplyRankingAnnotationWithPost(TaskTTLDedupeWithPost):
    DEDUPE_CACHE = TTLCache(maxsize=10_000, ttl=60)
    DEDUPE_NAME = "reply ranking"


class TaskRateLimitCoordinatedSpamAnnotationWithPost(TaskTTLDedupeWithPost):
    DEDUPE_CACHE = TTLCache(maxsize=10_000, ttl=60)
    DEDUPE_NAME = "coordinated spam"
