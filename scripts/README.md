# Scripts Directory

Вспомогательные скрипты для работы с проектом.

## Доступные скрипты

### train_model.py
Обучение модели и сохранение артефактов.

```bash
cd scripts
python3 train_model.py
```

Скрипт:
1. Загружает данные из `../data/raw/train.csv`
2. Выполняет препроцессинг
3. Обучает RandomForestClassifier
4. Сохраняет модель в `../models/model_artifacts.joblib`

### test_api.py
Тестирование API эндпоинтов.

```bash
# Сначала запустите API сервер
cd ..
uvicorn app.main:app --port 8080

# В другом терминале запустите тесты
cd scripts
python3 test_api.py
```

Скрипт тестирует:
- `/health` - проверка здоровья
- `/` - информация о модели
- `/predict` - одиночное предсказание
- `/predict_batch` - пакетные предсказания
