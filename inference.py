import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel, PeftConfig

def load_and_generate_answer(model_path: str, question: str) -> str:
    """
    Загружает модель с LoRA и генерирует ответ на вопрос.

    Args:
        model_path (str): Путь к папке с моделью
        question (str): Вопрос для модели.

    Returns:
        str: Сгенерированный ответ.
    """

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    config = PeftConfig.from_pretrained(model_path)
    base_model = GPT2LMHeadModel.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, model_path)

    # Устройство (GPU или CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Подготовка ввода
    input_text = f"Вопрос: {question} Дай конкретный и полезный ответ:"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True, padding="max_length").to(device)

    # Генерация
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Дай конкретный и полезный ответ:")[-1].strip()
    return answer

# Пример использования
if __name__ == "__main__":
    model_path = r"C:\Users\Mi\PycharmProjects\Final_project\best_rugpt3_lora_2"
    question = "Как записаться на курсы в университете?"
    answer = load_and_generate_answer(model_path, question)
    print(f"Вопрос: {question}")
    print(f"Ответ: {answer}")