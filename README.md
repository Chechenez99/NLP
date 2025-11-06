# NLP

Проект для генерации реалистичных банковских диалогов и классификации интентов (16 категорий: переводы, блокировка карты, курс валют, выписка и т.д.).

# Что делает проект

Генерирует короткие диалоги user ↔ assistant на русском под заданный интент
(генератор: Qwen/Qwen2.5-3B-Instruct, 8-бит через bitsandbytes).

Очищает и валидирует данные (русский текст, ≥3 реплики, есть user: и assistant:, дедуп по хэшу).

Обучает классификатор интентов
(модель: DeepPavlov/rubert-base-cased + Trainer).

Выдаёт предсказание интента и подтягивает шаблонный ответ из словаря CANNED_ANSWERS.

Строит classification report и confusion matrix.

# Данные

Формат: JSONL с полями {"label": "<intent>", "dialog": "<text>"}.

Папки по умолчанию: data_bank_dialogs/train.jsonl, data_bank_dialogs/test.jsonl.

Разбиение: train / valid (stratify) / test.

# Модели и настройки

Генерация: Qwen/Qwen2.5-3B-Instruct (pipeline "text-generation", temperature/top-p/top-k, бан-лист метаслов).

Классификация: DeepPavlov/rubert-base-cased (AutoModelForSequenceClassification),
TrainingArguments: lr=3e-5, epochs=4, per_device_train_bs=8, eval/save=epoch,
metric_for_best_model=f1_macro, gradient_checkpointing, bf16/fp16 при доступности.

# Метрики

Accuracy (evaluate: "accuracy").

F1 (macro/weighted) (evaluate: "f1").

sklearn.classification_report по всем 16 интентам.

Confusion matrix (Matplotlib).

# Технологии

Python, PyTorch, Transformers, evaluate, scikit-learn, NumPy, Matplotlib, bitsandbytes.
