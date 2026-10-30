"""asyncio.gather with a progress bar."""

import asyncio

from tqdm.auto import tqdm


async def gather(coros, desc: str) -> list:
    """Run coroutines concurrently and return their results in order, updating a progress bar."""
    coros = list(coros)
    bar = tqdm(total=len(coros), desc=desc, leave=False)

    async def tick(c):
        r = await c
        bar.update(1)
        return r

    try:
        return list(await asyncio.gather(*[tick(c) for c in coros]))
    finally:
        bar.close()
