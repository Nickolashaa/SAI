# Лабораторные работы по СИИ

| Папка | Работа | Запуск |
| --- | --- | --- |
| `1/` | Поиск лиц и глаз с веб-камеры (каскады Хаара) | `uv run lab1` |
| `2/` | Распознавание объекта по устойчивым признакам (SIFT) | `uv run lab2 образец.jpg` |
| `3/` | Контурный анализ и распознавание текста (Tesseract) | `uv run lab3 contours ...` / `uv run lab3 ocr ...` |

Python-зависимости в каждой папке ставятся своим `uv sync`. Ниже — то, что
`uv` поставить не может: системные пакеты.

## Системные зависимости

### Tesseract OCR — только для ЛР 3, часть 2

Без него `lab3 ocr` не запустится; `lab3 contours` и остальные работы
Tesseract не используют.

```bash
# Arch / CachyOS
sudo pacman -S --needed tesseract tesseract-data-rus tesseract-data-eng

# Debian / Ubuntu
sudo apt install tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng

# Fedora
sudo dnf install tesseract tesseract-langpack-rus tesseract-langpack-eng
```

Проверка:

```bash
tesseract --version
tesseract --list-langs     # в списке должны быть rus и eng
```

На Windows — установщик с https://github.com/UB-Mannheim/tesseract/wiki
(отметить русский язык при установке), затем путь передаётся флагом:

```bash
uv run lab3 ocr картинка.png --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### Шрифт с кириллицей — для ЛР 3

Русский текст на изображение выводится через Pillow, поэтому нужен TTF-шрифт
с кириллицей. В большинстве систем он уже стоит; если подписи пропадают:

```bash
# Arch / CachyOS
sudo pacman -S --needed noto-fonts ttf-dejavu fontconfig

# Debian / Ubuntu
sudo apt install fonts-noto-core fonts-dejavu-core fontconfig
```

Программа ищет `NotoSans-Regular.ttf` и `DejaVuSans.ttf`, а если не находит —
спрашивает шрифт у `fc-match` из `fontconfig`.

### Веб-камера — для ЛР 1 и ЛР 2

Отдельных пакетов не нужно, камера читается через V4L2 средствами OpenCV.
Устройство должно быть видно в системе:

```bash
ls /dev/video*
```

Если устройств несколько, нужное выбирается флагом `--camera N`.

## Что ставить не нужно

OpenCV, NumPy, Pillow и pytesseract ставятся через `uv sync` в каждой папке.
Каскады Хаара для ЛР 1 лежат прямо в репозитории (`1/src/lab1/cascades/`) —
скачивать их отдельно не требуется. Кодеки для записи видео в mp4 входят в
колесо `opencv-python`, системный ffmpeg не нужен.

Во всех работах используется `opencv-python` версии 4.x: в OpenCV 5 удалён
класс `CascadeClassifier`, на котором построена ЛР 1.
