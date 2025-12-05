# train.py - نسخه نهایی با نمایش دقت لحظه‌ای + بدون خطا

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import torch
import re
import urllib.parse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

def preprocess(url):
    url = str(url).lower()
    url = urllib.parse.unquote(url)
    p = urllib.parse.urlparse(url)
    domain = re.sub(r'^www\.', '', p.netloc.split(':')[0])
    path = p.path or "/"
    query = "?" + p.query if p.query else ""
    return f"[DOMAIN] {domain} [PATH] {path} [QUERY] {query}"

print("در حال لود دیتاست...")
df = pd.read_csv('data/malicious_phish.csv')[['url', 'type']].dropna()

print("توزیع اولیه:")
print(df['type'].value_counts())

# تقسیم
train_val, test = train_test_split(df, test_size=0.15, random_state=42, stratify=df['type'])
train, val = train_test_split(train_val, test_size=0.15, random_state=42, stratify=train_val['type'])

# بالانس train
benign = train[train['type'] == 'benign']
others = train[train['type'] != 'benign']
benign_down = benign.sample(n=len(others)//3 + 12000, random_state=42, replace=False)
train_balanced = pd.concat([benign_down, others]).sample(frac=1, random_state=42)

# لیبل
le = LabelEncoder()
train_balanced['label'] = le.fit_transform(train_balanced['type'])
val['label'] = le.transform(val['type'])
test['label'] = le.transform(test['type'])

# پیش‌پردازش
train_balanced['text'] = train_balanced['url'].apply(preprocess)
val['text'] = val['url'].apply(preprocess)
test['text'] = test['url'].apply(preprocess)

# حل مشکل __index_level_0__
train_balanced = train_balanced.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)

# دیتاست
dataset = DatasetDict({
    'train': Dataset.from_pandas(train_balanced[['text', 'label']], preserve_index=False),
    'val': Dataset.from_pandas(val[['text', 'label']], preserve_index=False),
    'test': Dataset.from_pandas(test[['text', 'label']], preserve_index=False)
})

# توکنایزر و مدل
tokenizer = AutoTokenizer.from_pretrained("jackaduma/SecBERT")

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, max_length=256)

tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.remove_columns(['text'])
tokenized = tokenized.rename_column('label', 'labels')
tokenized.set_format('torch')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained("jackaduma/SecBERT", num_labels=4)

# این تابع دقت و F1 رو در هر اِپوک نشون میده
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

# تنظیمات آموزش + نمایش دقت
args = TrainingArguments(
    output_dir="./secbert_result",
    num_train_epochs=6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=20,
    eval_steps=500,                    # هر 500 قدم ارزیابی کنه
    evaluation_strategy="steps",       # به جای epoch → هر چند قدم دقت رو نشون بده
    save_strategy="steps",
    save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",  # بهترین مدل بر اساس دقت ذخیره بشه
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=2,
    report_to="none",
    seed=42,
    disable_tqdm=False,
    logging_dir='./logs',
    logging_first_step=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['val'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,   # این خط دقت رو نشون میده!
)


trainer.train()

# ارزیابی نهایی روی تست
print("\nارزیابی نهایی روی تست...")
preds = trainer.predict(tokenized['test'])
y_pred = np.argmax(preds.predictions, axis=1)
print("\n" + classification_report(preds.label_ids, y_pred, target_names=le.classes_, digits=6))

# ذخیره نهایی
trainer.save_model("./final_phishing_model_995")
tokenizer.save_pretrained("./final_phishing_model_995")
print("\nمدل نهایی  ذخیره شد!")