# Описание Результатов Проекта (к 14 октября 2024)

## Обзор

В ходе летнего проекта была изучена возможность модификации архитектуры трансформера для выполнения задач регрессии. 
Исследование сосредоточилось на возможности дублирования слоев во время инференса и 
определении влияния такой модификации на точность и производительность модели.

## Основные Результаты

### Дублирование Слоев в Трансформере

- **Гипотеза**: Проверить, можно ли дублировать слои трансформера для улучшения моделей, обученных на задачи линейной регрессии.
- **Результаты**: Дублирование слоев на этапе инференса не показало положительных результатов. Даже при перемешивании слоев, эффективность модели не улучшалась. Это говорит о том, что слои отличаются по своим функциям и, вероятно, решают различные независимые подзадачи, способствующие улучшению итогового результата. 
- **Вывод**: Слои трансформера, хотя и используют, казалось бы, одинаковый подход к оптимизации, судя по данным linear probing (см. статью "Transformers Learn Higher-Order Optimization Methods for In-Context Learning: A Study with Linear Models."), не взаимозаменяемы.
   Повторное применение одного слоя может негативно сказываться на уже оптимизированных компонентах эмбеддинга, хотя как выделить эти компоненты для изучения пока неясно.

### Использование RNN-Подобной Архитектуры

- **Гипотеза**: Применение архитектуры, схожей с рекуррентной нейронной сетью (RNN), использующей многоразовое применение небольшого набора слоев будет работать также как использование того же числа независимых слоев.
- **Результаты**: Повторное использование трех слоев в формате RNN, показало значительное улучшение графиков in-context-learning (ICL) и функции потерь (loss) по сравнению с использованием шести независимых слоев.
- **Заключение**: Данная методика продемонстрировала свою перспективность, но требует дальнейшей оптимизации и доработки.

## Заключение

В проекте было проведено значительное количество экспериментов, касающихся внутренних механизмов работы и оптимизации трансформеров. Несмотря на отсутствие ожидаемых улучшений при дублировании слоев, 
были найдены новые подходы (как в случае использования RNN-структур), которые могут послужить основой для дальнейших исследований.

## Планы на Будущее

- Дальнейшее изучение и оптимизация подхода с использованием RNN-структуры в трансформерах.
- Поиск и выделение компонент эмбеддинга, оптимизируемых отдельными слоями.
- Эксперименты с другими модификациями архитектуры для повышения точности и производительности моделей.
