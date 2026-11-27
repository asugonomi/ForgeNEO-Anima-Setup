# ForgeNEO Anima セットアップ

Windows x64 向けの日本語 Forge Neo 環境です。最初に [Install-ForgeNEO.bat](https://github.com/asugonomi/ForgeNEO-Anima-Setup/releases/latest/download/Install-ForgeNEO.bat) を保存し、十分な空き容量のあるフォルダーでダブルクリックしてください。

BATと同じ場所に ForgeNEO-Anima フォルダーができます。セットアップ後は ForgeNEO.exe で起動し、ForgeNEO-Update.exe で更新します。コンソールの文字を選択してコピーでき、ログは logs に保存されます。

公式 Anima Base v1.0、Qwenテキストエンコーダー、VAEは [CircleStone Labs](https://huggingface.co/circlestone-labs/Anima) から取得します。FnMoChIrMrgBl5_int8ConvRot は [kurousagi132の公開モデル](https://huggingface.co/kurousagi132/FnMoChIrMrgBl5_int8ConvRot) から自動で取得し、初期モデルとして選択します。いずれもAPIキーは不要です。

## FnMoの配布元登録

リポジトリの models.json の url に、公開Hugging Faceファイルの https://huggingface.co/所有者/モデル名/resolve/コミット/ファイル名 形式のURLを指定します。ダウンロード時に使用中の原本のサイズとSHA-256を照合します。1台だけに設定する場合は ForgeNEO-Update.exe の「6」でファイルのURLを登録し、「3」で同期してください。

## 構成

使用中のForge本体、外部拡張30件と有効・無効設定、ワイルドカード、日本語翻訳、タグ辞書、プリセット、ADetailer・SAM・アップスケーラーを復元します。内容は inventory.json で確認できます。追加のLoRAやワイルドカードが参照する個別LoRAは、必要に応じて forge/models/Lora に置いてください。

Windows x64、インターネット接続、NTFSのローカルドライブを想定しています。初回は空き容量35 GiB以上を推奨します。GPU用PyTorchの導入後に演算を試し、利用できるバックエンドを選びます。CPUでは生成に時間とメモリーが必要です。機種やドライバーによって、個別拡張やINT8演算の対応状況が異なります。

コードとモデルの利用条件はそれぞれの配布元に従ってください。

## 更新と復元

Forgeを終了してから ForgeNEO-Update.exe を起動し、「1」で本体、「2」で外部拡張を更新します。GitHub管理の21拡張は各リポジトリを確認し、ローカル由来の9拡張はこのリポジトリの local-extensions を確認します。変更対象がユーザー編集と競合した場合は、そのコンポーネントを保留し、ファイル名を表示します。依存パッケージの更新は専用のPython環境で試し、PyTorchの演算確認を通過してから切り替えます。「5」で直前の更新を1件ずつ戻せます。

セットアップを中断した場合は、同じ場所からBATを実行してください。取得済みのモデルはハッシュ照合で再利用し、ダウンロード途中のファイルは再開します。Pythonの仮想環境がインストール先を記録するため、移設時は新しい場所でBATを実行してください。

通常のログは ForgeNEO-Anima/logs にあります。起動EXEはコンソールに同じログを表示します。別ポートでの起動には環境変数 FORGENEO_PORT を、ブラウザーを自動で開かない起動には FORGENEO_NO_BROWSER=1 を使用できます。

初回はモデルとソフトウェアを合わせて十数GBを取得します。システムドライブの空き容量が少ない場合は、別ドライブのフォルダーにBATを置いてください。セットアップに使う一時ファイルとHugging Faceキャッシュも導入先に作成します。

## 実機で確認した範囲

Windows x64のRTX 4070 Ti SUPER（CUDA）と、同じPCのCore i7-9700KF（CPU専用PyTorch）で、起動EXEからGradioを開き、Anima Base v1.0とFnMoの両方で384×384の画像を生成しました。AMD ROCmとIntel XPUはGPUの自動検出経路を備えていますが、該当GPUでの生成は未検証です。既存環境とポートが重なるスマートフォン連携拡張2件は生成テスト時に無効化しました。配布時は元の有効・無効設定を使用します。

## GPU・CPUの選択

通常のダブルクリックではGPUで処理します。対応GPUが見つからない場合や演算テストが失敗した場合は停止し、ログに理由を表示します。CPU専用で使いたい場合だけ、コマンドプロンプトから `Install-ForgeNEO.bat --cpu` を実行してください。手動で選んだCPUは次回起動にも引き継ぎます。GPUへ戻すには更新EXEの「4」を選びます。

旧版を導入済みの場合は、Forgeを終了し、新しいBATを従来と同じ場所で実行してください。旧版が自動選択したCPUはGPUの再判定対象になります。プロンプトスタイル（styles.csv）はユーザー自身が管理します。配布するプリセットは config.json と ui-config.json です。

PythonとGitの事前導入は不要です。セットアップは専用のPython 3.13.15を展開し、既存のGitが見つからなければMinGitを自動取得します。
