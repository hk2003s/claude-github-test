# Whisper × LiveKit

LiveKit ルームに参加している各参加者の音声を、ローカルで動く Whisper（faster-whisper）で文字起こしするワーカーです。
文字起こし結果は、ルーム内へデータメッセージ（トピック `transcript`、JSON）として配信されます。

## 構成

| ファイル | 役割 |
|---|---|
| `agent.py` | LiveKit に参加し、音声トラックを購読して一定長ごとに文字起こしを配信 |
| `transcriber.py` | faster-whisper のラッパー（VAD フィルタ付き） |
| `requirements.txt` | 依存ライブラリ |
| `.env.example` | 設定例（コピーして `.env` を作成） |

## セットアップ

```bash
cd whisper-livekit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 値を編集
set -a && source .env && set +a
python agent.py
```

ワーカーを起動したあと、同じ `LIVEKIT_ROOM` に別のクライアント（LiveKit Playground など）から
マイクをオンにして参加すると、そのトラックの文字起こしが始まります。

## 受信側の例（JavaScript）

```js
room.on(RoomEvent.DataReceived, (payload, participant, kind, topic) => {
  if (topic !== 'transcript') return;
  const msg = JSON.parse(new TextDecoder().decode(payload));
  console.log(msg.identity, msg.text);
});
```

## 注意点

- 固定長（`CHUNK_SECONDS`、既定 5 秒）で区切るため、文の途中で切れることがあります。精度を上げるには VAD ベースの区切りへの変更を検討してください。
- `small` モデル + `int8` は CPU で動作しますが、リアルタイム性は CPU 性能に依存します。GPU があれば `WHISPER_DEVICE=cuda` と `WHISPER_COMPUTE_TYPE=float16` を推奨します。
- 本リポジトリ内では LiveKit サーバーに接続しての実行テストは行っていません（構文・API シグネチャのみ確認）。
