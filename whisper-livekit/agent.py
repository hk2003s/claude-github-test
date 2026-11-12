"""LiveKit worker that transcribes every remote audio track with Whisper.

Each finished chunk is published back to the room as a data message on the
topic "transcript" with a JSON payload:
  {"identity": "...", "text": "...", "language": "en", "timestamp": 1731330000.0}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

import numpy as np
from livekit import rtc
from livekit.api import AccessToken, VideoGrants

from transcriber import Transcriber

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("whisper-livekit")

SAMPLE_RATE = 16_000
CHUNK_SECONDS = float(os.getenv("CHUNK_SECONDS", "5"))
TRANSCRIPT_TOPIC = "transcript"


def make_bot_token(room_name: str, identity: str) -> str:
    key = os.environ["LIVEKIT_API_KEY"]
    secret = os.environ["LIVEKIT_API_SECRET"]
    return (
        AccessToken(key, secret)
        .with_identity(identity)
        .with_name("Whisper Transcriber")
        .with_grants(
            VideoGrants(room_join=True, room=room_name, can_publish_data=True)
        )
        .to_jwt()
    )


async def transcribe_track(
    room: rtc.Room,
    participant: rtc.RemoteParticipant,
    track: rtc.RemoteAudioTrack,
    transcriber: Transcriber,
) -> None:
    stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1)
    chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
    buffer: list[np.ndarray] = []
    buffered = 0

    async def flush() -> None:
        nonlocal buffer, buffered
        if not buffered:
            return
        audio = np.concatenate(buffer).astype(np.float32) / 32768.0
        buffer, buffered = [], 0
        result = await asyncio.to_thread(transcriber.transcribe, audio)
        if result is None:
            return
        payload = json.dumps(
            {
                "identity": participant.identity,
                "text": result.text,
                "language": result.language,
                "timestamp": time.time(),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        await room.local_participant.publish_data(payload, topic=TRANSCRIPT_TOPIC)
        log.info("[%s] %s", participant.identity, result.text)

    try:
        async for event in stream:
            frame = event.frame
            samples = np.frombuffer(frame.data, dtype=np.int16)
            buffer.append(samples.copy())
            buffered += samples.size
            if buffered >= chunk_samples:
                await flush()
        await flush()  # track ended: transcribe the remainder
    finally:
        await stream.aclose()


async def main() -> None:
    url = os.environ["LIVEKIT_URL"]
    room_name = os.environ["LIVEKIT_ROOM"]
    identity = os.getenv("BOT_IDENTITY", "whisper-bot")

    transcriber = Transcriber()
    room = rtc.Room()
    tasks: dict[str, asyncio.Task] = {}

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        log.info("subscribed to audio from %s", participant.identity)
        tasks[publication.sid] = asyncio.create_task(
            transcribe_track(room, participant, track, transcriber)
        )

    @room.on("track_unsubscribed")
    def on_track_unsubscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        task = tasks.pop(publication.sid, None)
        if task:
            task.cancel()

    await room.connect(url, make_bot_token(room_name, identity))
    log.info("connected to room %s as %s", room_name, identity)
    try:
        await asyncio.Event().wait()  # run until interrupted
    finally:
        for task in tasks.values():
            task.cancel()
        await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
