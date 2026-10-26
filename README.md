# Dictée

Petit outil de dictée vocale **100 % local**, pensé pour un PC Windows sans GPU.
On enregistre depuis le micro dans une page web locale, la transcription
apparaît dans une zone de texte modifiable, et on la copie où on veut
(bouton de copie intégré).

- Français surtout, anglais parfois, détection automatique de la langue.
- Ponctuation et majuscules incluses.
- Aucun service cloud : le modèle tourne sur le CPU, la page n'écoute que sur `127.0.0.1`.

Deux modèles au choix :

| Modèle | Moteur | Vocabulaire personnalisé |
|---|---|---|
| **Parakeet TDT 0.6B v3** (par défaut) | ONNX Runtime, int8 (`onnx-asr`) | non |
| **Whisper large-v3-turbo** | CTranslate2, int8 (`faster-whisper`) | oui (champ « Vocabulary / jargon ») |

## Installation sous Windows

1. Installer `uv` (sans droits administrateur), dans PowerShell :

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Puis fermer et rouvrir le terminal.

2. Récupérer le projet (clone git ou copie du dossier) :

   ```powershell
   git clone <url-du-depot> dictee
   cd dictee
   ```

3. Installer les dépendances (uv télécharge Python 3.12 si besoin) :

   ```powershell
   uv sync
   ```

4. Lancer :

   ```powershell
   uv run app.py
   ```

   Le navigateur s'ouvre sur <http://127.0.0.1:7860>.

## Utilisation

1. Choisir le modèle (Parakeet par défaut).
2. Cliquer sur **Enregistrer**, parler, puis **Arrêter**.
3. La transcription se lance automatiquement à l'arrêt. Le texte apparaît
   dans la zone « Transcription », modifiable, avec un bouton de copie en haut à droite.
4. La ligne d'état indique le modèle, la durée de l'audio et le temps de
   transcription, par ex. `Parakeet · 12.4 s audio · 1.8 s`.

**Autorisation du micro** : au premier enregistrement, le navigateur demande
l'accès au microphone. Il faut accepter. Si la demande a été refusée par erreur,
cliquer sur l'icône à gauche de l'adresse (cadenas ou réglages) et autoriser le micro pour
`127.0.0.1`. Les navigateurs autorisent le micro sur `127.0.0.1` même sans HTTPS.

Chaque modèle est chargé au premier usage puis gardé en mémoire : passer de
l'un à l'autre ne le recharge pas.

## Premier lancement : téléchargement des modèles

Au premier usage de chaque modèle, il est téléchargé depuis Hugging Face puis
mis en cache (`%USERPROFILE%\.cache\huggingface\hub`) :

| Modèle | Dépôt Hugging Face | Taille |
|---|---|---|
| Parakeet (int8) | `istupakov/parakeet-tdt-0.6b-v3-onnx` | ≈ 0,67 Go |
| Whisper turbo | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | ≈ 1,6 Go |

Le dépôt Whisper ne contient que des poids float16 (1,6 Go) ; la conversion en int8 se
fait au chargement, en mémoire.

## Utilisation hors ligne (`DICTEE_MODELS_DIR`)

Si le proxy de l'entreprise bloque Hugging Face, copier les modèles (par ex. par
clé USB) depuis une machine qui y a accès, puis définir la variable
`DICTEE_MODELS_DIR`. Les modèles sont alors lus uniquement dans ce dossier, sans
accès réseau. Arborescence attendue :

```
D:\modeles\                          ← DICTEE_MODELS_DIR
├── parakeet-tdt-0.6b-v3-onnx\
│   ├── config.json
│   ├── encoder-model.int8.onnx
│   ├── decoder_joint-model.int8.onnx
│   └── vocab.txt
└── faster-whisper-large-v3-turbo\
    ├── config.json
    ├── model.bin
    ├── preprocessor_config.json
    ├── tokenizer.json
    └── vocabulary.json
```

On n'est pas obligé de fournir les deux modèles : seul celui qu'on utilise doit être présent.

Pour récupérer ces fichiers sur la machine connectée :

```powershell
uvx --from huggingface_hub hf download istupakov/parakeet-tdt-0.6b-v3-onnx config.json encoder-model.int8.onnx decoder_joint-model.int8.onnx vocab.txt --local-dir parakeet-tdt-0.6b-v3-onnx
uvx --from huggingface_hub hf download mobiuslabsgmbh/faster-whisper-large-v3-turbo --local-dir faster-whisper-large-v3-turbo
```

Puis, sur le poste cible :

```powershell
$env:DICTEE_MODELS_DIR = "D:\modeles"
uv run app.py
```

(Pour rendre la variable permanente : `setx DICTEE_MODELS_DIR "D:\modeles"`, puis rouvrir le terminal.)

## Proxy d'entreprise

- `uv sync` et le téléchargement des modèles respectent les variables de proxy standard :

  ```powershell
  $env:HTTPS_PROXY = "http://proxy.entreprise:8080"
  ```

- Si l'entreprise dispose d'un miroir Hugging Face interne :

  ```powershell
  $env:HF_ENDPOINT = "https://miroir-hf.entreprise.local"
  ```

- Erreurs de certificat (proxy qui inspecte le TLS) : `uv sync --native-tls` utilise
  le magasin de certificats Windows.
- En dernier recours, passer en hors ligne avec `DICTEE_MODELS_DIR` (voir plus haut).

En cas d'échec de chargement, la ligne d'état affiche `⚠ Could not load …` et le
détail est dans le terminal.

## Quel modèle choisir ?

- **Parakeet (par défaut)** : beaucoup plus rapide sur CPU et très précis en
  français (WER ≈ 5 % sur FLEURS). À utiliser pour la dictée de tous les jours.
- **Whisper large-v3-turbo** : nettement plus lent sur CPU, mais accepte un
  **vocabulaire** (noms propres, acronymes, jargon métier) via le champ
  « Vocabulary / jargon », par ex. `Kubernetes, SharePoint, DSI, M. Dupont`.
  À choisir quand Parakeet écorche des termes spécifiques.

Ordre de grandeur mesuré sur un Mac (Apple Silicon, CPU uniquement) :

| Audio | Parakeet | Whisper |
|---|---|---|
| 5,6 s | 0,13 s | 9 s |
| 116 s | 2,9 s | 64 s |

Un portable Windows d'entreprise sera probablement plus lent.

## Tests

```powershell
uv run pytest           # tests rapides (conversion audio, cas limites)
uv run pytest -m slow   # transcription réelle du fichier tests/fr_fixture.wav avec les deux modèles
```
