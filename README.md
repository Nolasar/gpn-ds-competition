# gpn-ds-competition

Работа выполнена в рамка GPN Intelligence Cup 2024 по направлению "Data scientist в нефтяном ритейле".

# Поставновка задачи: 
1. Выделить подмножество тематик из корпуса цитат
2. Построить систему для определения тематики цитат по уже накопленной истории извлеченных тематик.

# Обзор решение:
В данном разделе представлено краткое описание идей, более подробное описание каждого пункта можно найти в файле `solution.ipynb`
1. Любая система начинается с исследования и обработки входных даннных, поэтому первым делом реализуется пайплайн предобработки, который включается в себя:
   - Токенизацию
   - Лемматизацию
   - Коррекцию ошибок
   - Удаление лишних символов
   - Кастомная обработка на основе наблюдений автора
2. Тематическое моделирование или выделение подмножества тематик. Основная идея заключается в трех этапах:
   - Генерация эмбеддингов для текстовых данных
   - Кластеризация с использованием полученных эмбеддингов
   - Генерация тематик для каждого кластера на основе подмножества цитат, входящих в данный кластер
     
   Для удобства используется библиотека `BERTopic`, которая автоматизирует вышеописанный процесс и дает возможность для тонкой настройки   моделей.
  
3. Классификация цитат по тематикам
  - В пункте 2. было получено подмножество тематик и каждая цитата была сопоставлена с соотвествующей тематикой. При наличии размеченного датасета появляется возможность применять методы обучения с учителем.
  - В качестве *бейзлайн* модели была выбрана модель `RandomForestClassifier` из библиотеки `sklearn`
  - Поскольку модель показана высокие показатели метрик, например, `accuracy >0.95` было принято решение оставить ее в качестве основной

# Допущения
Одним из серьезных допущений является тот факт, что корпус цитат однозначно разделяется на кластеры, что, в общем-то, не так. Одна цитата может относиться сразу к нескольким тематикам, что не учитывается в данной работе.

В пункте 2. стоит считать вероятности попадания цитаты в каждый из кластеров и отбирать проходящие заранее заданный порог (гиперпараметр)

В таком случае в пункте 3. необходимо использовать модель, поддерживающую классификацию вида "многие ко многим"

При учете вышеописанных допущений и советов, расписанных в `solution.ipynb`, данный подход к решению задачи считаю корректным.

# Запуск программы
Установка зависимостей
```
pip install -r requirements.txt
```
ВАЖНО! При запуске кода необходимо стабильное соединение с интернетом, т.к. используется `API` `Yandex.Speller`. При использовании некоторых `VPN` может выдавать ошибку соединения с `Hugging Face` при скачивании преобученных моделей.
