<p align="left">
        <a href="README.md">English</a> &nbsp｜ &nbsp <a href="README_CN.md">中文</a> &nbsp｜ &nbsp 日本語
</p>
<br><br>

<p align="center">
    <img src="assets/logo.png" width="400"/>
<p>

<p align="center">
<br>
<a href="https://qwenlm.github.io/blog/">ブログ</a>&nbsp&nbsp| &nbsp&nbsp<a href="https://arxiv.org/abs/2407.10759">論文</a>&nbsp&nbsp | &nbsp&nbsp&nbsp<a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat</a>&nbsp&nbsp | &nbsp&nbsp<a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp</a>
</p>
<br><br>

Qwen-Audioの最新の進展であるQwen2-Audioを紹介します。Qwen2-Audioは、大規模な音声言語モデルであり、さまざまな音声信号入力を受け入れ、音声指示に基づいて音声分析や直接的なテキスト応答を行うことができます。2つの異なる音声インタラクションモードを紹介します：

* 音声チャット：ユーザーはテキスト入力なしでQwen2-Audioと自由に音声で対話できます。

* 音声分析：ユーザーはインタラクション中に音声とテキスト指示を提供して音声を分析できます。

**Qwen2-Audioシリーズの2つのモデル、Qwen2-AudioとQwen2-Audio-Chatを近日中にリリース予定です。**

## アーキテクチャ

Qwen2-Audioの3段階のトレーニングプロセスの概要。

<p align="left">
    <img src="assets/framework.png" width="80%"/>
<p>

## ニュースと更新
* 2024.7.15 🎉 **Qwen2-Audio**の論文を発表し、関連するモデル構造、トレーニング方法、およびモデルのパフォーマンスを紹介しました。詳細は[レポート](https://arxiv.org/abs/2407.10759)をご覧ください！

* 2023.11.30 🔥 **Qwen-Audio**シリーズをリリースしました。

<br>

## 評価
Qwen2-Audioの能力を以下の13の標準ベンチマークで評価しました：
<table><thead><tr><th>タスク</th><th>説明</th><th>データセット</th><th>分割</th><th>メトリック</th></tr></thead><tbody><tr><td rowspan="4">ASR</td><td rowspan="4">自動音声認識</td><td>Fleurs</td><td>dev | test</td><td rowspan="4">WER</td></tr><tr><td>Aishell2</td><td>test</td></tr><tr><td>Librispeech</td><td>dev | test</td></tr><tr><td>Common Voice</td><td>dev | test</td></tr><tr><td>S2TT</td><td>音声からテキストへの翻訳</td><td>CoVoST2</td><td>test</td><td>BLEU </td></tr><tr><td>SER</td><td>音声感情認識</td><td>Meld</td><td>test</td><td>ACC</td></tr><tr><td>VSC</td><td>声の分類</td><td>VocalSound</td><td>test</td><td>ACC</td></tr><tr><td rowspan="4">AIR-Bench<br></td><td>チャットベンチマーク-音声</td><td>Fisher<br>SpokenWOZ<br>IEMOCAP<br>Common voice</td><td>dev | test</td><td>GPT-4 Eval</td></tr><tr><td>チャットベンチマーク-音</td><td>Clotho</td><td>dev | test</td><td>GPT-4 Eval</td></tr>
<tr><td>チャットベンチマーク-音楽</td><td>MusicCaps</td><td>dev | test</td><td>GPT-4 Eval</td></tr><tr><td>チャットベンチマーク-混合音声</td><td>Common voice<br>AudioCaps<br>MusicCaps</td><td>dev | test</td><td>GPT-4 Eval</td></tr></tbody></table>

以下は全体のパフォーマンスです：
<p align="left">
    <img src="assets/radar_compare_qwen_audio.png" width="70%"/>
<p>

評価の詳細は以下の通りです：
<br>
<b>（注：ここで示す評価結果は、元のトレーニングフレームワークの初期モデルに基づいています。ただし、フレームワークをHuggingfaceに変換した後、スコアが低下しました。ここでは、論文の初期モデル結果から始めて、完全な評価結果を示します。）</b>

<table><thead><tr><th rowspan="2">タスク</th><th rowspan="2">データセット</th><th rowspan="2">モデル</th><th colspan="2">パフォーマンス</th></tr><tr><th>メトリック</th><th>結果</th></tr></thead><tbody><tr><td rowspan="15">ASR</td><td rowspan="7"><b>Librispeech</b><br>dev-clean | dev-other | <br>test-clean | test-other</td><td>SpeechT5</td><td rowspan="7">WER </td><td>2.1 | 5.5 | 2.4 | 5.8</td></tr><tr><td>SpeechNet</td><td>- | - | 30.7 | -</td></tr><tr><td>SLM-FT</td><td>- | - | 2.6 | 5.0</td></tr><tr><td>SALMONN</td><td>- | - | 2.1 | 4.9</td></tr><tr><td>SpeechVerse</td><td>- | - | 2.1 | 4.4</td></tr><tr><td>Qwen-Audio</td><td>1.8 | 4.0 | 2.0 | 4.2</td></tr><tr><td>Qwen2-Audio</td><td><b>1.3 | 3.4 | 1.6 | 3.6</b></td></tr><tr><td rowspan="2"><b>Common Voice 15</b> <br>en | zh | yue | fr</td><td>Whisper-large-v3</td><td rowspan="2">WER </td><td>9.3 | 12.8 | 10.9 | 10.8</td></tr><tr><td>Qwen2-Audio</td><td><b>8.6 | 6.9 | 5.9 | 9.6</b></td></tr>
<tr><td rowspan="2"><b>Fleurs</b> <br>zh</td><td>Whisper-large-v3</td><td rowspan="2">WER </td><td>7.7</td></tr><tr><td>Qwen2-Audio</td><td><b>7.5</b></td></tr><tr><td rowspan="4"><b>Aishell2</b> <br>Mic | iOS | Android</td><td>MMSpeech-base</td><td rowspan="4">WER </td><td>4.5 | 3.9 | 4.0</td></tr><tr><td>Paraformer-large</td><td>- | <b>2.9</b> | -</td></tr><tr><td>Qwen-Audio</td><td>3.3 | 3.1 | 3.3</td></tr><tr><td>Qwen2-Audio</td><td><b>3.0</b> | 3.0 | <b>2.9</b></td></tr><tr><td rowspan="8">S2TT</td><td rowspan="5"><b>CoVoST2</b> <br>en-de | de-en | <br>en-zh | zh-en</td><td>SALMONN</td><td rowspan="5">BLEU </td><td>18.6 | - | 33.1 | -</td></tr><tr><td>SpeechLLaMA</td><td>- | 27.1 | - | 12.3</td></tr><tr><td>BLSP</td><td>14.1 | - | - | -</td></tr><tr><td>Qwen-Audio</td><td>25.1 | 33.9 | 41.5 | 15.7</td></tr><tr><td>Qwen2-Audio</td><td><b>29.9 | 35.2 | 45.2 | 24.4</b></td></tr>
<tr><td rowspan="3"><b>CoVoST2</b> <br>es-en | fr-en | it-en |</td><td>SpeechLLaMA</td><td rowspan="3">BLEU </td><td>27.9 | 25.2 | 25.9</td></tr><tr><td>Qwen-Audio</td><td>39.7 | <b>38.5</b> | 36.0</td></tr><tr><td>Qwen2-Audio</td><td><b>40.0 | 38.5 | 36.3</b></td></tr><tr><td rowspan="3">SER</td><td rowspan="3"><b>Meld</b></td><td>WavLM-large</td><td rowspan="3">ACC </td><td>0.542</td></tr><tr><td>Qwen-Audio</td><td><b>0.557</b></td></tr><tr><td>Qwen2-Audio</td><td>0.553</td></tr><tr><td rowspan="4">VSC</td><td rowspan="4"><b>VocalSound</b></td><td>CLAP</td><td rowspan="4">ACC </td><td>0.4945</td></tr><tr><td>Pengi</td><td>0.6035</td></tr><tr><td>Qwen-Audio</td><td>0.9289</td></tr><tr><td>Qwen2-Audio</td><td><b>0.9392</b></td></tr>
<tr><td>AIR-Bench <br></td><td><b>チャットベンチマーク</b><br>音声 | 音 |<br> 音楽 | 混合音声</td><td>SALMONN<br>BLSP<br>Pandagpt<br>Macaw-LLM<br>SpeechGPT<br>Next-gpt<br>Qwen-Audio<br>Gemini-1.5-pro<br>Qwen2-Audio</td><td>GPT-4 </td><td>6.16 | 6.28 | 5.95 | 6.08<br>6.17 | 5.55 | 5.08 | 5.33<br>3.58 | 5.46 | 5.06 | 4.25<br>0.97 | 1.01 | 0.91 | 1.01<br>1.57 | 0.95 | 0.95 | 4.13<br>3.86 | 4.76 | 4.18 | 4.13<br>6.47 | 6.95 | 5.52 | 6.08<br>6.97 | 5.49 | 5.06 | 5.27<br><b>7.18 | 6.99 | 6.79 | 6.77</b></td></tr></tbody></table>

<b>（次にHuggingfaceに変換した後）</b>

<table><thead><tr><th rowspan="2">タスク</th><th rowspan="2">データセット</th><th rowspan="2">モデル</th><th colspan="2">パフォーマンス</th></tr><tr><th>メトリック</th><th>結果</th></tr></thead><tbody><tr><td rowspan="15">ASR</td><td rowspan="7"><b>Librispeech</b><br>dev-clean | dev-other | <br>test-clean | test-other</td><td>SpeechT5</td><td rowspan="7">WER </td><td>2.1 | 5.5 | 2.4 | 5.8</td></tr><tr><td>SpeechNet</td><td>- | - | 30.7 | -</td></tr><tr><td>SLM-FT</td><td>- | - | 2.6 | 5.0</td></tr><tr><td>SALMONN</td><td>- | - | 2.1 | 4.9</td></tr><tr><td>SpeechVerse</td><td>- | - | 2.1 | 4.4</td></tr><tr><td>Qwen-Audio</td><td>1.8 | 4.0 | 2.0 | 4.2</td></tr><tr><td>Qwen2-Audio</td><td><b>1.7 | 3.6 | 1.7 | 4.0</b></td></tr><tr><td rowspan="2"><b>Common Voice 15</b> <br>en | zh | yue | fr</td><td>Whisper-large-v3</td><td rowspan="2">WER </td><td>9.3 | 12.8 | 10.9 | 10.8</td></tr><tr><td>Qwen2-Audio</td><td><b>8.7 | 6.5 | 5.9 | 9.6</b></td></tr>
<tr><td rowspan="2"><b>Fleurs</b> <br>zh</td><td>Whisper-large-v3</td><td rowspan="2">WER </td><td>7.7</td></tr><tr><td>Qwen2-Audio</td><td><b>7.0</b></td></tr><tr><td rowspan="4"><b>Aishell2</b> <br>Mic | iOS | Android</td><td>MMSpeech-base</td><td rowspan="4">WER </td><td>4.5 | 3.9 | 4.0</td></tr><tr><td>Paraformer-large</td><td>- | <b>2.9</b> | -</td></tr><tr><td>Qwen-Audio</td><td>3.3 | 3.1 | 3.3</td></tr><tr><td>Qwen2-Audio</td><td><b>3.2</b> | 3.1 | <b>2.9</b></td></tr><tr><td rowspan="8">S2TT</td><td rowspan="5"><b>CoVoST2</b> <br>en-de | de-en | <br>en-zh | zh-en</td><td>SALMONN</td><td rowspan="5">BLEU </td><td>18.6 | - | 33.1 | -</td></tr><tr><td>SpeechLLaMA</td><td>- | 27.1 | - | 12.3</td></tr><tr><td>BLSP</td><td>14.1 | - | - | -</td></tr><tr><td>Qwen-Audio</td><td>25.1 | <b>33.9</b> | 41.5 | 15.7</td></tr><tr><td>Qwen2-Audio</td><td><b>29.6</b> | 33.6 | <b>45.6</b> | <b>24.0</b></td></tr>
<tr><td rowspan="3"><b>CoVoST2</b> <br>es-en | fr-en | it-en |</td><td>SpeechLLaMA</td><td rowspan="3">BLEU </td><td>27.9 | 25.2 | 25.9</td></tr><tr><td>Qwen-Audio</td><td><b>39.7 | 38.5 | 36.0</b></td></tr><tr><td>Qwen2-Audio</td><td>38.7 | 37.2 | 35.2</td></tr><tr><td rowspan="3">SER</td><td rowspan="3"><b>Meld</b></td><td>WavLM-large</td><td rowspan="3">ACC </td><td>0.542</td></tr><tr><td>Qwen-Audio</td><td><b>0.557</b></td></tr><tr><td>Qwen2-Audio</td><td>0.535</td></tr><tr><td rowspan="4">VSC</td><td rowspan="4"><b>VocalSound</b></td><td>CLAP</td><td rowspan="4">ACC </td><td>0.4945</td></tr><tr><td>Pengi</td><td>0.6035</td></tr><tr><td>Qwen-Audio</td><td>0.9289</td></tr><tr><td>Qwen2-Audio</td><td><b>0.9395</b></td></tr>
<tr><td>AIR-Bench <br></td><td><b>チャットベンチマーク</b><br>音声 | 音 |<br> 音楽 | 混合音声</td><td>SALMONN<br>BLSP<br>Pandagpt<br>Macaw-LLM<br>SpeechGPT<br>Next-gpt<br>Qwen-Audio<br>Gemini-1.5-pro<br>Qwen2-Audio</td><td>GPT-4 </td><td>6.16 | 6.28 | 5.95 | 6.08<br>6.17 | 5.55 | 5.08 | 5.33<br>3.58 | 5.46 | 5.06 | 4.25<br>0.97 | 1.01 | 0.91 | 1.01<br>1.57 | 0.95 | 0.95 | 4.13<br>3.86 | 4.76 | 4.18 | 4.13<br>6.47 | <b>6.95</b> | 5.52 | 6.08<br>6.97 | 5.49 | 5.06 | 5.27<br><b>7.24</b> | 6.83 | <b>6.73</b> | <b>6.42</b></td></tr></tbody></table>

すべての**評価スクリプト**を提供し、結果を再現します。**Huggingface、ModelScope、Web UI**を近日中に提供します。
数日お待ちください。プロセスを進めています。

## デモ
より印象的なケースは、[Qwenのブログ](https://qwenlm.github.io/blog/)で更新されます。

## 採用情報

フルタイムまたはインターンとして参加することに興味がある場合は、qwen_audio@list.alibaba-inc.comまでご連絡ください。
<br>

## ライセンス契約

各モデルのHFリポジトリ内のライセンスを確認してください。商業利用の申請を提出する必要はありません。
<br>

## 引用

私たちの論文やコードがあなたの研究に役立つ場合は、スター :star: と引用 :pencil: を検討してください :)

```BibTeX
@article{Qwen-Audio,
  title={Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models},
  author={Chu, Yunfei and Xu, Jin and Zhou, Xiaohuan and Yang, Qian and Zhang, Shiliang and Yan, Zhijie  and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2311.07919},
  year={2023}
}
```

```BibTeX
@article{Qwen2-Audio,
  title={Qwen2-Audio Technical Report},
  author={Chu, Yunfei and Xu, Jin and Yang, Qian and Wei, Haojie and Wei, Xipin and Guo,  Zhifang and Leng, Yichong and Lv, Yuanjun and He, Jinzheng and Lin, Junyang and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2407.10759},
  year={2024}
}
```
<br>

## お問い合わせ

研究チームまたは製品チームにメッセージを残したい場合は、qianwen_opensource@alibabacloud.comまでメールでご連絡ください。
