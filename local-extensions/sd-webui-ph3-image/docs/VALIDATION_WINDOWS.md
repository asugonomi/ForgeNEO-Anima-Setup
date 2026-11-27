# Windows実機検証 2026-11-05

0.1.0-experimentalへのローカル互換修正。元のVALIDATION.mdは配布時点のCPU検証記録として保持する。

## 直接原因と修正

実Forgeの生成コールバックで `AttributeError: 'list' object has no attribute 'keys'` を再現した。
公式tokenizer_config.jsonのextra_special_tokensはリストだが、導入済みTransformers 4.57.6は同項目に名前付き辞書を要求する。
トークナイザー構築で停止し、モデル推論に到達していなかった。配布版の診断はファイル存在だけを確認していたため成功していた。

- ph3_native/tokenizer.py: リストをadditional_special_tokensに渡す互換アダプター。辞書形式はそのまま扱い、元ファイルは変更しない。
- ph3_native/engine.py: 上記アダプターを使用。
- ph3_native/diagnostics.py: 実トークナイザーの構築と短文エンコードも検査。
- tests/test_tokenizer.py: 特殊トークン保持、辞書形式の非変更、実tokenizer.jsonとのID一致を検査。
- ph3_native/ui.py: この環境の実機確認済み範囲を表示。

## 実環境

- Windows 11、RTX 4070 Ti SUPER 16GB、RAM 63.94GiB。
- Forge Neo 2.31、commit 7d6fd36c58a0e861640bcfe421e4f990921d7f2a。本体の既存独自変更は保持。
- 本体: E:\EasyForgeNeo\sd-webui-forge-neo
- 起動: 画面に表示されるCMDから E:\EasyForgeNeo\ForgeNeo_Fast.bat --skip-prepare-environment
- Fastの --uv --pin-shared-memory --cuda-malloc --cuda-stream を使用。追加フラグは今回の検証起動だけの依存更新防止用。
- Python: 本体venv\Scripts\python.exe、3.13.15。
- Torch 2.11.0+cu130、CUDA build 13.0、Gradio 4.40.0、Transformers 4.57.6、comfy-kitchen 0.2.40、safetensors 0.8.0。
- 固定native commit: 9387d9189273384f14143707898126abbcb7ae74。
- 本体models\P-H3-Image内の既存W4A8 DiT／INT8 text encoder／INT8静止画VAEを再利用。モデルの追加取得なし。
- ComfyUIの導入・起動・サーバー委譲なし。Torch等の共有依存変更なし。

## 実生成

2048×2048、1枚、Seed 123456、固定8ステップ、作業用VRAM6GiB、VAEタイル256、CPUキャッシュ保持。
プロンプト: A photograph of a red ceramic teapot on a wooden table beside a sunny window, green plants in the background, natural soft light.

| 計測 | 初回 | 2回目 |
|---|---:|---:|
| 生成本体 | 57.56秒 | 44.97秒 |
| コールバック（保存・後始末含む） | 59.80秒 | 46.34秒 |
| Torch allocatedピーク | 8.55GiB | 8.55GiB |
| Torch reservedピーク | 8.78GiB | 8.69GiB |
| Windows専用GPUメモリ・観測最大 | 9.02GiB | 8.94GiB |
| Windows共有GPUメモリ・観測最大 | 0.446GiB | 0.433GiB |
| Forge RSS・観測最大 | 23.38GiB | 23.38GiB |
| PC全体使用RAM・観測最大 | 43.70GiB | 41.38GiB |

Windowsカウンターは約1.12〜1.41秒間隔。瞬間ピークの保証はない。Torch計測値とWindowsのプロセス別GPUカウンターは定義が異なる。
2回目はCPUモデル・文章埋め込みキャッシュを再利用。初回もOSファイルキャッシュの完全コールド状態ではない。

出力: G:\Stable_diffusion_outputs\txt2img-images\p-h3-image\2026-11-05\

- 225521_123456_f8295548.png / .json
- 225654_123456_7cd21664.png / .json

PNG整合性、2048×2048 RGB、メタデータを検査。2枚は画素単位で一致。
赤いティーポット、木のテーブル、窓辺、植物を目視確認。目立つタイル境界、黒画像、異常色は見られなかった。
参照実装との数値・画質一致を証明するものではない。

## メモリ・停止・復帰

- 実重みの厳格ロード、INT8文章エンコード、W4A8 DiT、VAEタイル復元、PNG保存が通過。
- DiTは約8009MiBをGPUに部分ロードし、約4733MiBをCPU側に残した。前段のGPUモデルを退避してVAEへ切り替えた。
- DiT処理中とVAE処理中に拡張の停止を実行し、例外による終了と後続の解放を確認。その後に実生成成功。
- VAE欄にDiTファイルを渡す負のテストでは、DiT推論後に形式エラーを返し、PNGを作らず終了。
- キャッシュ保持OFFでエラー後に専用GPU約0.372GiB、共有約0.082GiB、RSS約2.37GiB以下へ戻ったことを観測。CUDAコンテキスト分は残る。
- 同じForgeプロセスの通常txt2imgでchosenIrisesMix_v20Anima-2.9B-graft-int8を生成。1024×1024、20ステップ、Euler/Simple、CFG4、Seed123456。18.0秒。
- 通常生成画像は G:\Stable_diffusion_outputs\txt2img-images\2026-11-05\00003-123456.png。目視確認済み。既存Anima NativeのTeaCacheとLayer Replayも動作。
- 通常生成検査ではその呼び出しだけHiresとADetailerをOFFにし、設定保存は行っていない。

## 制限

CPUテストはForgeのPython 3.13.15と作業フォルダーに隔離したpytestで69件成功（13.49秒）。Forge環境へpytestをインストールしていない。
最終コードの再起動後、ブラウザー上のP-H3タブと診断ボタンからapi_and_headers_ok=true、tokenizer_encode_ok=true、errors=[]を確認した。
config.jsonの内容とui-config.jsonのハッシュは作業開始前と同一。本体git statusも開始時と同じ。

検証途中のForge停止はエージェントが再起動目的で行ったもの。CPUキャッシュ解放のクラッシュではない。
エージェントの再起動手順のミス後、ユーザーが可視CMDでFast.batを起動し直した。最終確認ではそのプロセスを終了していない。

実CUDA OOMは発生していないため、OOM再試行・解放は既存CPU模擬テストの範囲。GPUメモリを無理に枯渇させる試験は未実施。
文章エンコード途中での停止、2912×2912、長時間連続生成、参照実装との比較、全拡張の全組み合わせは未検証。
ComfyUIと同一速度・同一VRAM使用量とは主張しない。実験版表記は維持する。
