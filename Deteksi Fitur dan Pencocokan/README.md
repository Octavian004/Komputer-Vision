# FeatureMatch Vision App

Aplikasi komprehensif untuk deteksi fitur, pencocokan, pengenalan objek, dan image retrieval.

## Struktur
- `main.py`: entry point
- `detector_benchmark.py`: Fase 1
- `robust_matcher.py`: Fase 2
- `object_recognizer.py`: Fase 3
- `image_retriever.py`: Fase 4
- `utils/`: helper utilities
- `data/`: template, database, query
- `results/`: output PNG dan log

## Cara Jalankan
1. install dependencies:

```bash
pip install -r requirements.txt
```

2. jalankan modul:

```bash
python main.py --benchmark
python main.py --matcher
python main.py --recognize
python main.py --retrieve
```

## Catatan
- Pastikan isi data di `data/templates`, `data/database`, `data/query` dengan minimal 10 gambar.
- Konfigurasi parameter mudah diedit di `main.py`.
