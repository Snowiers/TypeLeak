import asyncio
import threading
import unittest

from exposure.classifier import CallableSource, QueueSource
from exposure.event import Prediction

TOPK = [("e", 0.94), ("o", 0.02), ("t", 0.02), ("g", 0.01), ("a", 0.01)]


def prediction(key="e"):
    return Prediction(key_topk=[(key, 0.94)] + TOPK[1:])


async def take(source, count):
    collected = []
    async for item in source.predictions():
        collected.append(item)
        if len(collected) == count:
            break
    return collected


class TestQueueSource(unittest.TestCase):
    def test_yields_submitted_predictions_in_order(self):
        source = QueueSource()

        async def scenario():
            for key in "abc":
                source.submit_nowait(prediction(key))
            return await take(source, 3)

        result = asyncio.run(scenario())
        self.assertEqual([p.key_topk[0][0] for p in result], ["a", "b", "c"])

    def test_submit_before_loop_starts_is_buffered(self):
        # The model side may hand us predictions before the server is running.
        source = QueueSource()
        source.submit(prediction("z"))
        result = asyncio.run(take(source, 1))
        self.assertEqual(result[0].key_topk[0][0], "z")

    def test_close_ends_the_stream(self):
        source = QueueSource()

        async def scenario():
            source.submit_nowait(prediction("a"))
            source.close()
            collected = []
            async for item in source.predictions():
                collected.append(item)
            return collected

        self.assertEqual(len(asyncio.run(scenario())), 1)

    def test_backlog_is_bounded_and_drops_oldest(self):
        # A live exposure monitor should shed stale keystrokes rather than grow without
        # bound; a current keystroke is worth more than a queued old one.
        source = QueueSource(max_backlog=3)

        async def scenario():
            for key in "abcdef":
                source.submit_nowait(prediction(key))
            return await take(source, 3)

        result = asyncio.run(scenario())
        self.assertEqual([p.key_topk[0][0] for p in result], ["d", "e", "f"])
        self.assertEqual(source.dropped, 3)

    def test_submit_from_another_thread(self):
        source = QueueSource()

        async def scenario():
            # Bind the loop first, the way EventServer's pump would.
            iterator = source.predictions()
            asyncio.get_running_loop()
            source._loop = asyncio.get_running_loop()
            thread = threading.Thread(target=lambda: source.submit(prediction("t")))
            thread.start()
            thread.join()
            return await anext(iterator)

        result = asyncio.run(scenario())
        self.assertEqual(result.key_topk[0][0], "t")


class TestCallableSource(unittest.TestCase):
    def test_pulls_from_a_sync_function(self):
        keys = iter("abc")
        source = CallableSource(lambda: prediction(next(keys, None)) if True else None)

        async def scenario():
            return await take(source, 3)

        result = asyncio.run(scenario())
        self.assertEqual(len(result), 3)

    def test_pulls_from_an_async_function(self):
        remaining = ["a", "b"]

        async def produce():
            return prediction(remaining.pop(0)) if remaining else None

        source = CallableSource(produce)
        result = asyncio.run(take(source, 2))
        self.assertEqual([p.key_topk[0][0] for p in result], ["a", "b"])

    def test_none_ends_the_stream(self):
        source = CallableSource(lambda: None)

        async def scenario():
            collected = []
            async for item in source.predictions():
                collected.append(item)
            return collected

        self.assertEqual(asyncio.run(scenario()), [])


if __name__ == "__main__":
    unittest.main()
