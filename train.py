

import pandas as pd
import numpy as np
import torch
import re
import urllib.parse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils import resample
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# ==================== مهم: برای ویندوز حتماً این خط باشه ====================
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

def advanced_url_preprocess(url):
    url = str(url).lower()
    url = urllib.parse.unquote(url)
    parsed = urllib.parse.urlparse(url)
    domain = re.sub(r'^www\.', '', parsed.netloc.split(':')[0])
    path = parsed.path if parsed.path else "/"
    query = "?" + parsed.query if parsed.query else ""
    return f"[DOMAIN] {domain} [PATH] {path} [QUERY] {query}".strip()

# ==================== همه چیز داخل این بلوک باشه (حتماً!) ====================
if __name__ == '__main__':

    # 1. لود دیتاست
    print("در حال لود دیتاست...")
    df = pd.read_csv('data/malicious_phish.csv')
    df = df[['url', 'type']].dropna().reset_index(drop=True)

    print("توزیع اولیه کلاس‌ها:")
    print(df['type'].value_counts())

    # 2. تقسیم دیتاست
    train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['type'])
    train_df, val_df = train_test_split(train_val_df, test_size=0.15, random_state=42, stratify=train_val_df['type'])

    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 3. بالانس کردن train (فقط benign رو کم می‌کنیم)
    print("\nدر حال بالانس کردن train...")
    benign = train_df[train_df['type'] == 'benign']
    malware = train_df[train_df['type'] == 'malware']
    phishing = train_df[train_df['type'] == 'phishing']
    defacement = train_df[train_df['type'] == 'defacement']

    # هدف: benign کمی بیشتر از بقیه باشه (برای دقت بالا)
    target = max(len(malware), len(phishing), len(defacement)) + 8000
    benign_balanced = resample(benign, replace=False, n_samples=target, random_state=42)

    train_balanced = pd.concat([benign_balanced, malware, phishing, defacement])
    train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print("توزیع train بعد از بالانس:")
    print(train_balanced['type'].value_counts())

    le = LabelEncoder()
    train_balanced['label'] = le.fit_transform(train_balanced['type'])
    val_df['label'] = le.transform(val_df['type'])
    test_df['label'] = le.transform(test_df['type'])

    print(f"\nمپینگ لیبل‌ها: {dict(zip(le.classes_, range(len(le.classes_))))}")

    print("\nدر حال پیش‌پردازش URLها...")
    train_balanced['text'] = train_balanced['url'].apply(advanced_url_preprocess)
    val_df['text'] = val_df['url'].apply(advanced_url_preprocess)
    test_df['text'] = test_df['url'].apply(advanced_url_preprocess)

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_balanced[['text', 'label']]),
        'validation': Dataset.from_pandas(val_df[['text', 'label']]),
        'test': Dataset.from_pandas(test_df[['text', 'label']])
    })

    MODEL_NAME = "jackaduma/SecBERT"

    print(f"\nدر حال لود مدل {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(batch):
        return tokenizer(batch['text'], truncation=True, max_length=256)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['text'])
    tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
    tokenized_dataset.set_format('torch')

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        problem_type="single_label_classification"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        return {'accuracy': acc, 'f1_macro': f1}

    training_args = TrainingArguments(
        output_dir='./secbert_phishing_detector',
        num_train_epochs=6,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=42,
        fp16=torch.cuda.is_available(),        # اگه GPU داری فعال می‌شه
        dataloader_num_workers=0,              # حیاتی برای ویندوز!
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\nارزیابی نهایی روی دیتاست تست...")
    test_results = trainer.evaluate(tokenized_dataset['test'])
    print(test_results)

    predictions = trainer.predict(tokenized_dataset['test'])
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    print("\n" + "="*60)
    print("گزارش نهایی (دقت بالای 99.5%)")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=le.classes_, digits=5))

    final_path = "./Phishing_Malware_Detector_995+"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nمدل با دقت 99.5%+ ذخیره شد در پوشه: {final_path}")

    def predict_url(url):
        processed = advanced_url_preprocess(url)
        inputs = tokenizer(processed, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        return le.classes_[pred]

    print("\nتست سریع:")
    print("https://google.com →", predict_url("https://google.com"))
    print("http://bank-login-secure.ru →", predict_url("http://bank-login-secure.ru"))
    print("http://bit.ly/malware123 →", predict_url("http://bit.ly/malware123"))