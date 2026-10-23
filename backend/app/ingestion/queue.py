from typing import Protocol
from uuid import UUID

from arq import create_pool
from arq.connections import RedisSettings
from redis.exceptions import RedisError


class JobQueue(Protocol):
    async def enqueue(self, job_id: UUID) -> None: ...


class QueueUnavailable(RuntimeError):
    pass


class ArqJobQueue:
    def __init__(self, redis_url: str) -> None:
        self.redis_settings = RedisSettings.from_dsn(redis_url)

    async def enqueue(self, job_id: UUID) -> None:
        try:
            pool = await create_pool(self.redis_settings)
        except (OSError, RedisError) as error:
            raise QueueUnavailable("Could not enqueue video ingestion job") from error
        try:
            await pool.enqueue_job("process_video", str(job_id), _job_id=str(job_id))
        except (OSError, RedisError) as error:
            raise QueueUnavailable("Could not enqueue video ingestion job") from error
        finally:
            await pool.close()
