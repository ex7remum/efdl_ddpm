# Task 1
Найденные ошибки:

1) После запуска наших тестов на тесте *test_unet* появилась ошибка, что в
unet.py в 111 строчке ошибка при броадкастинге. Вывел рамер тензоров =>
у *thro* он был [2, 256, 1, 1], а у *temb* [2, 256] => добавил .view(),
чтобы былв нужная размерность, после этого тест стал проходить.

2) Дальше стал падать последний assert *test_diffusion*,
 соответственно ошибка должна была быть либо в forward() в ddpm, либо
 в формулах при подсчете коэффицентов для диффузии. При просмотре формул
 была найдена ошибка, что в forward() перед eps стоял не тот коэффицент:
 должно было быть self.sqrt_one_minus_alpha_prod, а не self.one_minus_alpha_over_prod.
 
 3) Дальше, все равно продолжал падать тест, но теперь я заметил, что 
 лосс получается слишком маленький, относительно того, который нужен был,
 значит что-то не так с лоссом или его компонентами и так я заметил, что 
 eps генерируется, как rand(), а не как randn(), как нужно в статье.
 
 4) Дальше тест стал то проходить, то падать и для воспроизводимости
 в начале теста был зафиксирован сид генерации.
 
 5) В *test_train_on_one_batch* падал тест с gpu, 
 поэтому во всех местах, где это нужно было добавлено 
 .to(device). 
 
 6) Реализовал *test_training*, который просто прогоняет весь пайплайн и
 контролирует, что ничего не упало. Для ускорения берется всего 3 эпохи,
 а также в качестве выборки берется 8 случайных элементов их исходной
 выборки. Так как 
 я потом еще дописывал кое-что в исходный код, чтобы
 все работало с wandb, то
 coverage получился 84%.
 
 7) После этого все тесты стали проходить, но генерируемые 
 картинки после ~10 эпох стали получаться слишком черными.
 Поэтому при сохранении сгенерированных картинок, стал,
 во-первых клипать их, чтобы они находились в диапазоне
 от -1 до 1, и в make_grid явно стал указывать, что
 картинки в диапазоне от -1 до 1 находятся.
 

# Task 2

Добавил в функции в *training.py* параметр *is_logging*, который отвечает
за то логгируем мы данные в wandb или нет (чтобы и тесты корректно работали и
потом все правильно запускалось). Логгируется конфиг, лосс, картинки
из первого батча на каждой эпоххе и сгенерированные картинки.

Ссылка на запуск в wandb (только wandb без hydra и dvc):

https://wandb.ai/extremum/ddpm_efdl/runs/sytnv1bp?workspace=user-extremum

# Task 3

Добавлены hydra-конфиги, чтобы изменять все гиперпараметры при обучении, а
также чтобы выбирать оптимизатор и то, использовать ли флипы в качесте
 аугментаций при обучении.  

Ссылка на оригинальный запуск:

https://wandb.ai/extremum/ddpm_efdl/runs/cgpnd5r8?workspace=user-extremum

Ссылки на измененные эксперименты (были сделаны после добавления dvc).

SGD + lr = 1e-4:

https://wandb.ai/extremum/ddpm_efdl/runs/m392l2u7?workspace=user-extremum

Маленькая модель:

https://wandb.ai/extremum/ddpm_efdl/runs/njyc83tk?workspace=user-extremum

RandomHorizontalFlip + lr=1e-3(так как сид фиксирован, то можно увидеть, что
на одинаковых эпохах картинки флипнуты относительно прошлых ранов):

https://wandb.ai/extremum/ddpm_efdl/runs/naxa4mh3?workspace=user-extremum





# Task 4

Добавлен весь пайплайн с dvc. Пайплайн состоит из двух частей: подготовка 
датасета (он просто скачивается :) ) вынесена в *get_data.py* и 
непосредствено запуск *main.py* после этого. То есть теперь мы в *main.py*
ничего не скачиваем. 

Итого, чтобы запустить весь пайплайн обучения нужно сделать

```
dvc init

dvc repro
```

Чтобы через консоль изменять параметры экспериментов в hydra-конфигах можно
воспользоваться:

https://dvc.org/doc/user-guide/experiment-management/hydra-composition
  
